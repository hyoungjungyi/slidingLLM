"""WikiText-2 calibration / evaluation data for the LLM sliding-window experiments.

The datasets live in a pre-populated HuggingFace `datasets` cache at
`--data_root` (default `/home/kangeunjeon/geontack_kairi/data`), so everything
here runs offline: `wikitext2/` is passed as `cache_dir` to `load_dataset`.

Tokenizing the whole of wikitext-2 takes ~30s, so the concatenated token stream
is memoized under `<data_root>/../slidingllm_cache` (overridable) keyed by
tokenizer + split.
"""
import os
import hashlib

import torch

DEFAULT_DATA_ROOT = "/home/kangeunjeon/geontack_kairi/data"

# Sub-directory of `data_root` holding the HF cache for each dataset.
_CACHE_SUBDIR = {
    "wikitext2": "wikitext2",
    "ptb": "ptb",
}


def _token_cache_path(cache_dir, dataset, split, tokenizer):
    tag = getattr(tokenizer, "name_or_path", "tok")
    key = hashlib.md5(f"{tag}|{dataset}|{split}".encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"tokens_{dataset}_{split}_{key}.pt")


def _load_split_text(dataset, split, data_root):
    """Return the raw text of one split, joined the same way every SVD-LLM /
    Dobi-SVD style baseline joins it (`"\\n\\n".join(...)`)."""
    from datasets import load_dataset

    sub = _CACHE_SUBDIR.get(dataset)
    if sub is None:
        raise ValueError(f"unsupported dataset: {dataset}")
    cache_dir = os.path.join(data_root, sub)

    if dataset == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir, trust_remote_code=True)
        return "\n\n".join(ds["text"])
    if dataset == "ptb":
        hf_split = "validation" if split == "test" else split
        ds = load_dataset("ptb_text_only", "penn_treebank", split=hf_split, cache_dir=cache_dir, trust_remote_code=True)
        return "\n\n".join(ds["sentence"])
    raise ValueError(dataset)


def get_token_stream(tokenizer, dataset="wikitext2", split="train",
                     data_root=DEFAULT_DATA_ROOT, cache_dir=None):
    """Whole split as a single flat 1-D LongTensor of token ids."""
    cache_dir = cache_dir or os.path.join(data_root, "slidingllm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = _token_cache_path(cache_dir, dataset, split, tokenizer)
    if os.path.exists(path):
        return torch.load(path)

    text = _load_split_text(dataset, split, data_root)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    torch.save(ids, path)
    return ids


def get_calib_input_ids(tokenizer, nsamples=128, seqlen=2048, seed=42,
                        dataset="wikitext2", data_root=DEFAULT_DATA_ROOT):
    """`nsamples` random `seqlen`-token windows from the train split.

    Returns a LongTensor of shape (nsamples, seqlen). This is the standard
    GPTQ/SVD-LLM calibration sampler, so calibration data is identical across
    our method and both baselines.
    """
    import random

    ids = get_token_stream(tokenizer, dataset, "train", data_root)
    rng = random.Random(seed)
    n_tok = ids.shape[0]
    out = []
    for _ in range(nsamples):
        i = rng.randint(0, n_tok - seqlen - 1)
        out.append(ids[i:i + seqlen])
    return torch.stack(out)


def get_test_input_ids(tokenizer, seqlen=2048, dataset="wikitext2",
                       data_root=DEFAULT_DATA_ROOT):
    """Non-overlapping `seqlen` chunks covering the test split.

    Returns a LongTensor of shape (n_chunks, seqlen) — the usual wikitext-2
    perplexity protocol (trailing tokens that do not fill a chunk are dropped).
    """
    ids = get_token_stream(tokenizer, dataset, "test", data_root)
    n = ids.shape[0] // seqlen
    return ids[: n * seqlen].view(n, seqlen)
