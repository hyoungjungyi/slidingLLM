"""Shared LLM plumbing for the sliding-window compression experiments.

Everything in here is deliberately *layer-local*: the decoder layers are kept on
CPU and streamed to the GPU a few at a time, with hidden states cached on CPU.
That is what lets the sliding-window stages run on a single 40GB card even
though the rank-search student is ~1.6x the size of the dense model.
"""
import json
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

# The seven Linear layers per decoder block that we decompose. `lm_head` and the
# embeddings are left untouched (same convention as Dobi-SVD and SVD-LLM).
TARGET_LINEARS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# --------------------------------------------------------------------------
# model / module helpers
# --------------------------------------------------------------------------
def load_llm(model_id, dtype="bfloat16", seqlen=2048):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    model.eval()
    model.config.use_cache = False
    # The tensor-parallel code path in HF's LlamaAttention reaches into
    # `q_proj.weight`, which does not exist once we swap in a factored module.
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1
    model.seqlen = seqlen
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tokenizer


def get_layers(model):
    """The forward-order list of decoder blocks (the 'blocks' of our ViT code)."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):  # OPT
        return inner.decoder.layers
    raise ValueError(f"cannot locate decoder layers on {type(model).__name__}")


def get_embed_tokens(model):
    inner = getattr(model, "model", model)
    if hasattr(inner, "embed_tokens"):
        return inner.embed_tokens
    return inner.decoder.embed_tokens


def get_final_norm(model):
    inner = getattr(model, "model", model)
    return getattr(inner, "norm", None) or getattr(inner, "final_layer_norm", None)


def get_module(root, path):
    m = root
    for p in path.split("."):
        m = getattr(m, p)
    return m


def set_module(root, path, new):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def layer_linear_paths(layer):
    """Target linear paths that actually exist on this block."""
    out = []
    for p in TARGET_LINEARS:
        try:
            get_module(layer, p)
        except AttributeError:
            continue
        out.append(p)
    return out


# --------------------------------------------------------------------------
# decoder-layer forward plumbing
# --------------------------------------------------------------------------
def build_decoder_kwargs(seqlen, dtype, device, batch_size=1):
    """The (attention_mask, position_ids) every LlamaDecoderLayer forward needs.

    We only ever feed full, unpadded `seqlen` windows, so the mask is the plain
    causal mask and can be built once and reused.
    """
    try:
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        dummy = torch.zeros(batch_size, seqlen, 1, dtype=dtype, device=device)
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (batch_size, seqlen), dummy, 0)
    except Exception:
        min_val = torch.finfo(dtype).min
        mask = torch.full((seqlen, seqlen), min_val, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        attention_mask = mask[None, None].expand(batch_size, 1, seqlen, seqlen)
    position_ids = torch.arange(seqlen, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seqlen)
    return attention_mask, position_ids


def run_layer(layer, hidden, attention_mask, position_ids):
    """Call one decoder block, tolerating the tuple / dataclass return shapes."""
    out = layer(hidden, attention_mask=attention_mask, position_ids=position_ids)
    return out[0] if isinstance(out, (tuple, list)) else out


def run_layers(layers, hidden, attention_mask, position_ids):
    for layer in layers:
        hidden = run_layer(layer, hidden, attention_mask, position_ids)
    return hidden


# --------------------------------------------------------------------------
# hidden-state cache (CPU-resident activations at one layer boundary)
# --------------------------------------------------------------------------
class HiddenCache:
    """(nsamples, seqlen, hidden) activations held on CPU, iterated in batches."""

    def __init__(self, tensor):
        self.data = tensor  # CPU tensor

    @classmethod
    def from_embeddings(cls, model, input_ids, device):
        embed = get_embed_tokens(model).to(device)
        outs = []
        with torch.no_grad():
            for i in range(input_ids.shape[0]):
                outs.append(embed(input_ids[i:i + 1].to(device)).cpu())
        get_embed_tokens(model).to("cpu")
        return cls(torch.cat(outs, dim=0))

    def __len__(self):
        return self.data.shape[0]

    def batches(self, batch_size):
        for i in range(0, len(self), batch_size):
            yield self.data[i:i + batch_size]

    def advance(self, layer, device, batch_size, attention_mask_fn, desc=None):
        """Push every cached sample through `layer` and return the new cache."""
        outs = torch.empty_like(self.data)
        with torch.no_grad():
            it = range(0, len(self), batch_size)
            if desc:
                it = tqdm(it, desc=desc, leave=False)
            for i in it:
                x = self.data[i:i + batch_size].to(device)
                am, pid = attention_mask_fn(x.shape[0])
                outs[i:i + batch_size] = run_layer(layer, x, am, pid).cpu()
        return HiddenCache(outs)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_ppl(model, test_ids, device="cuda", batch_size=1, desc="ppl"):
    """Standard wikitext-2 perplexity: mean token NLL over non-overlapping chunks."""
    model = model.to(device)
    model.eval()
    nll_sum = 0.0
    n_tok = 0
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    for i in tqdm(range(0, test_ids.shape[0], batch_size), desc=desc):
        batch = test_ids[i:i + batch_size].to(device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = batch[:, 1:]
        nll_sum += loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).item()
        n_tok += shift_labels.numel()
    return float(torch.exp(torch.tensor(nll_sum / n_tok)))


# --------------------------------------------------------------------------
# parameter accounting
# --------------------------------------------------------------------------
def dense_linear_params(model):
    """Parameter count of the *original* (dense) target linears, from config."""
    cfg = model.config
    h, inter = cfg.hidden_size, cfg.intermediate_size
    kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head = h // cfg.num_attention_heads
    per_layer = h * h + 2 * (h * kv * head) + h * h + 3 * (h * inter)
    return per_layer * cfg.num_hidden_layers


def compressed_linear_params(model):
    """Parameter count of whatever currently occupies the target linear slots."""
    total = 0
    for layer in get_layers(model):
        for p in layer_linear_paths(layer):
            total += sum(x.numel() for x in get_module(layer, p).parameters())
    return total


def param_report(model):
    dense = dense_linear_params(model)
    comp = compressed_linear_params(model)
    total = sum(p.numel() for p in model.parameters())
    return {
        "linear_params": comp,
        "dense_linear_params": dense,
        "linear_param_ratio": comp / dense,
        "total_params": total,
    }


def layer_rank_string(model):
    """`ranks` of each factored layer, in forward order — for the CSV log."""
    ranks = []
    for layer in get_layers(model):
        for p in layer_linear_paths(layer):
            m = get_module(layer, p)
            r = getattr(m, "rank", None)
            if r is None and isinstance(m, nn.Sequential) and len(m) == 2:
                r = m[0].out_features
            ranks.append(str(int(r)) if r is not None else "full")
    return ",".join(ranks)


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def save_result(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[result] wrote {path}")


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t0


def seed_everything(seed):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
