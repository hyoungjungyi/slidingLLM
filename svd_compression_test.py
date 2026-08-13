"""svd_compression_test.py — our main method (svd_compression_sliding_train_rank.py)
with the STAGGERED even-odd Stage-2 window schedule grafted in from
svd_compression_stage2_even_odd.py.

Identical to svd_compression_sliding_train_rank.py except that the Stage-2
sliding-window rank search processes every OTHER window per epoch, alternating
parity (offset = epoch % 2). Stage-1 (global k search) and Stage-3 (sliding-window
fine-tuning) are unchanged.

Intended test configuration (set via CLI flags, see the run command):
  Stage 1: 20 epochs (global rank search)
  Stage 2: 30 epochs, window 4, staggered even-odd
  Stage 3:  4 epochs, window 4, stride 1
  temperature 20 (no decay), MSE loss only (no CE), no Cayley, no rel-MSE,
  no progressive window, calib 1024.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import csv, gc, math, random, argparse, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import timm
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from thop import profile
from tqdm import tqdm

# Real Dobi-SVD methodology (activation truncation + stable SVD + IPCA weight
# update) lives in the sibling script. The `--dobi_svd_sliding` path below reuses
# its rank-allocation stages verbatim so it is *identical* to dobisvd.py.
import dobisvd
from svd_compression_sliding import apply_v2_method, list_fc_paths, get_module

# ==========================================================================
# 0. Config & utilities
# ==========================================================================
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
CALIBRATION_SIZE = 1024
CALIBRATION_SEED = 42
GLOBAL_SEED = 42

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def model_name_for_filename(model_name):
    return model_name.replace("/", "_")

def _normalize_tokens(x):
    if x.dim() == 4:
        return x.reshape(x.shape[0], -1, x.shape[3])
    if x.dim() == 3:
        return x
    raise ValueError(f"Unexpected token tensor shape: {tuple(x.shape)}")

# ==========================================================================
# 1. Data loading
# ==========================================================================
class FlatImageNetValTimmLabels(Dataset):
    def __init__(self, val_dir: str, labels_txt: str, transform=None):
        self.val_dir = Path(val_dir)
        self.transform = transform
        self.images = sorted([p for p in self.val_dir.iterdir() if p.suffix.lower() in [".jpeg", ".jpg"]], key=lambda p: p.name)
        with open(labels_txt, "r") as f:
            self.labels = [int(x.strip()) for x in f if x.strip()]

    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]

def get_timm_dataloaders(
    batch_size=32,
    root=os.environ.get("IMAGENET_ROOT", "/home/kangeunjeon/geontack_kairi/data/imagenet"),
    model_name="deit_base_patch16_224",
    val_labels_filename="val_labels_timm.txt",
    num_workers=4,
    calibration_size=1024,
    calibration_seed=42,
):
    model = timm.create_model(model_name, pretrained=True)
    cfg = resolve_data_config({}, model=model)
    eval_transform = create_transform(**cfg, is_training=False)

    dataset_train = create_dataset("imagenet", root=root, split="train")
    loader_train = create_loader(
        dataset_train,
        input_size=cfg["input_size"],
        batch_size=batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=True,
        num_workers=num_workers,
    )

    val_dir = os.path.join(root, "val")
    val_labels_txt = os.path.join(root, val_labels_filename)
    if os.path.isdir(val_dir) and os.path.exists(val_labels_txt):
        dataset_eval = FlatImageNetValTimmLabels(val_dir, val_labels_txt, transform=eval_transform)
        loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        dataset_eval = create_dataset("imagenet", root=root, split="validation")
        loader_eval = create_loader(
            dataset_eval,
            input_size=cfg["input_size"],
            batch_size=batch_size,
            is_training=False,
            use_prefetcher=True,
            num_workers=num_workers,
        )

    g = torch.Generator()
    g.manual_seed(calibration_seed)
    indices = torch.randperm(len(dataset_train), generator=g).tolist()
    cali_indices = indices[:calibration_size]
    cali_dataset = Subset(dataset_train, cali_indices)
    
    cali_loader = create_loader(
        cali_dataset,
        input_size=cfg["input_size"],
        batch_size=batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=True,
        num_workers=num_workers,
    )

    return loader_train, loader_eval, dataset_train, dataset_eval, cali_loader

# ==========================================================================
# 2. Model introspection
# ==========================================================================
def count_gelu(m, x, y): m.total_ops += torch.DoubleTensor([int(y.numel())])
def count_layernorm(m, x, y): m.total_ops += torch.DoubleTensor([int(y.numel())])
def count_softmax(m, x, y): m.total_ops += torch.DoubleTensor([int(y.numel())])
def count_attention(m, x, y):
    b, n, c = x[0].shape
    num_heads = m.num_heads
    head_dim = c // num_heads
    qkv_macs = b * n * c * c * 3
    attn_macs = b * num_heads * n * head_dim * n
    out_macs = b * num_heads * n * n * head_dim
    proj_macs = b * n * c * c
    m.total_ops += torch.DoubleTensor([qkv_macs + attn_macs + out_macs + proj_macs])

def eval_macs(model_name, model, input_tensor, custom_ops):
    model.eval()
    macs, _ = profile(model, inputs=(input_tensor,), custom_ops=custom_ops, verbose=False)
    params = sum(p.numel() for p in model.parameters())
    return macs, params

def list_mlp_fc_paths(model):
    paths = []
    for name, module in model.named_modules():
        if type(module).__name__ in ['Linear', 'MaskedSVDLinear', 'DobiSVDLinear']:
            if "mlp" in name and ("fc1" in name or "fc2" in name):
                paths.append(name)
    return paths

def list_attn_paths(model, model_name):
    paths = []
    for name, module in model.named_modules():
        if type(module).__name__ in ['Linear', 'MaskedSVDLinear', 'DobiSVDLinear']:
            if "attn.qkv" in name or "attn.proj" in name:
                paths.append(name)
    return paths

def list_fc_paths(model, model_name):
    paths = list_mlp_fc_paths(model)
    paths.extend(list_attn_paths(model, model_name))
    return paths

def get_module(model, path):
    parts = path.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m

def get_parent_and_attr(model, path):
    parts = path.split(".")
    m = model
    for p in parts[:-1]:
        m = getattr(m, p)
    return m, parts[-1]

def replace_module(model, path, new_module):
    parent, attr = get_parent_and_attr(model, path)
    setattr(parent, attr, new_module)

def make_factored_linear(W_u, W_v, bias, dtype, device, use_cayley=False):
    r = W_u.shape[1]
    in_dim = W_v.shape[1]
    out_dim = W_u.shape[0]

    fc_v = nn.Linear(in_dim, r, bias=False).to(device=device, dtype=dtype)
    fc_u = nn.Linear(r, out_dim, bias=(bias is not None)).to(device=device, dtype=dtype)
    
    if use_cayley:
        # Split W_u into orthogonal directions and magnitudes (S)
        norm_u = torch.norm(W_u, dim=0, keepdim=True) # shape: (1, r)
        U_ortho = W_u / (norm_u + 1e-8)
        
        # We need a scaling layer in the middle
        fc_s = nn.Linear(r, r, bias=False).to(device=device, dtype=dtype)
        fc_s.weight.data = torch.diag(norm_u.squeeze(0)).to(dtype)
        
        fc_v.weight.data = W_v.to(dtype)
        fc_u.weight.data = U_ortho.to(dtype)
        if bias is not None:
            fc_u.bias.data = bias.to(dtype)
            
        import torch.nn.utils.parametrizations as P
        P.orthogonal(fc_u, "weight", orthogonal_map="cayley")
        
        return nn.Sequential(OrderedDict([
            ('fc_v', fc_v),
            ('fc_s', fc_s),
            ('fc_u', fc_u)
        ]))
    else:
        fc_v.weight.data = W_v.to(dtype)
        fc_u.weight.data = W_u.to(dtype)
        if bias is not None:
            fc_u.bias.data = bias.to(dtype)
        return nn.Sequential(OrderedDict([
            ('fc_v', fc_v),
            ('fc_u', fc_u)
        ]))

# ==========================================================================
# 3. Calibration
# ==========================================================================
def calibrate_stats_vit(model_name, cali_loader, device):
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    paths = list_fc_paths(model, model_name)

    cov = {}       
    n_samples = {}
    for p in paths:
        d = get_module(model, p).in_features
        cov[p] = torch.zeros(d, d, dtype=torch.float32, device=device)
        n_samples[p] = 0

    def make_hook(path):
        def hook(module, inputs):
            x = inputs[0].detach()
            x = _normalize_tokens(x) if x.dim() >= 3 else x
            x = x.reshape(-1, x.shape[-1]).to(torch.float32)
            b = x.shape[0]

            M_x = x.t() @ x
            cov[path] = cov[path] + M_x
            
            n_samples[path] += b

        return hook

    handles = []
    for p in paths:
        handles.append(get_module(model, p).register_forward_pre_hook(make_hook(p)))

    with torch.no_grad():
        for batch in tqdm(cali_loader, desc="Calibrating"):
            model(batch[0].to(device))

    for h in handles:
        h.remove()

    del model
    torch.cuda.empty_cache()

    for p in paths:
        cov[p] = cov[p] / n_samples[p]

    stats = {
        p: {"cov": cov[p].cpu(), "n": n_samples[p]}
        for p in paths
    }
    return stats

# ==========================================================================
# 4. Whitening methods
# ==========================================================================
def _whiten_from_eigs(V, L_new):
    L_new = torch.clamp(L_new, min=0.0)
    sqrt_L = torch.sqrt(L_new)
    inv_sqrt_L = torch.zeros_like(sqrt_L)
    mask = sqrt_L > 1e-12
    inv_sqrt_L[mask] = 1.0 / sqrt_L[mask]

    R = V * sqrt_L.unsqueeze(0)
    R_inv = (V * inv_sqrt_L.unsqueeze(0)).t()
    return R, R_inv

def _whiten(M, damp_alpha, device):
    d = M.shape[0]
    M = M.to(device, dtype=torch.float64)
    M = 0.5 * (M + M.t())
    
    diag_mean = torch.diag(M).mean()
    M = M + damp_alpha * diag_mean * torch.eye(d, dtype=torch.float64, device=device)
    
    L, V = torch.linalg.eigh(M)
    return _whiten_from_eigs(V, L)

def gptq_damp_whitening(M, device, damp_alpha=0.01):
    return _whiten(M, damp_alpha, device)

def build_whitening(whitening, M, n, device, damp_alpha=0.01):
    if whitening == "gptq_damp":
        return gptq_damp_whitening(M, device, damp_alpha=damp_alpha)
    else:
        raise ValueError(f"Unknown whitening method: {whitening}")

# ==========================================================================
# 5. Masked SVD Linear & Rank Search Setup
# ==========================================================================
class MaskedSVDLinear(nn.Module):
    def __init__(self, W_u, W_v, bias, dtype, device, init_ratio=0.5, path=None):
        super().__init__()
        # W_u: (out_dim, r_max)
        # W_v: (r_max, in_dim)
        self.W_u = nn.Parameter(W_u.to(dtype=dtype, device=device), requires_grad=False)
        self.W_v = nn.Parameter(W_v.to(dtype=dtype, device=device), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype=dtype, device=device), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        r_max = W_u.shape[1]
        
        if isinstance(init_ratio, dict):
            # Stage 2: Initialize smoothly around the boundaries learned in Stage 1
            o = W_u.shape[0]
            i = W_v.shape[1]
            K = init_ratio.get(path, max(1, round(0.5 * o * i / (o + i))))
            idx = torch.arange(r_max, device=device, dtype=torch.float32)
            z_init = (K - idx) * 0.1
        else:
            # Initialize smoothly around the uniform boundaries learned from init_ratio (float)
            o = W_u.shape[0]
            i = W_v.shape[1]
            ratio_val = init_ratio if isinstance(init_ratio, float) else 0.5
            K = max(1, round(ratio_val * o * i / (o + i)))
            idx = torch.arange(r_max, device=device, dtype=torch.float32)
            z_init = (K - idx) * 0.1
            
        self.Z = nn.Parameter(z_init)
        # High temperature for soft sigmoid gate
        self.temperature = 10.0 

    def get_mask(self):
        # Temperature-annealed Sigmoid gate
        mask = torch.sigmoid(self.Z / self.temperature)
        return mask

    def forward(self, x):
        mask = self.get_mask()
        # W_v: (r_max, in_dim). h = x @ W_v^T
        h = F.linear(x, self.W_v)
        h = h * mask
        # W_u: (out_dim, r_max). y = h @ W_u^T + bias
        y = F.linear(h, self.W_u, self.bias)
        return y

class DobiSVDLinear(nn.Module):
    def __init__(self, W_u, W_v, bias, dtype, device, init_ratio=0.5):
        super().__init__()
        self.W_u = nn.Parameter(W_u.to(dtype=dtype, device=device), requires_grad=False)
        self.W_v = nn.Parameter(W_v.to(dtype=dtype, device=device), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype=dtype, device=device), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        r_max = W_u.shape[1]
        o = W_u.shape[0]
        i = W_v.shape[1]
        target_k = max(1, round(init_ratio * o * i / (o + i)))
        self.k = nn.Parameter(torch.tensor(float(target_k), dtype=torch.float32, device=device))
        self.beta = 0.1 # Lower beta to provide a wide, soft gradient for k!
        self.register_buffer('i', torch.arange(r_max, device=device, dtype=torch.float32))

    def get_mask(self):
        mask = 0.5 * torch.tanh(self.beta * (self.k - self.i)) + 0.5
        return mask

    def forward(self, x):
        mask = self.get_mask()
        h = F.linear(x, self.W_v)
        h = h * mask
        y = F.linear(h, self.W_u, self.bias)
        return y

def apply_masked_svd(model_name, model, stats_cache, device, whitening="gptq_damp", damp_alpha=0.01, use_dobi=False, init_ratio=0.5):
    paths = list_fc_paths(model, model_name)

    R_cache = {}
    print(f"\n[Build Whitening] {whitening} ...")
    for p in tqdm(paths):
        n = stats_cache[p]["n"]
        M = stats_cache[p]["cov"].to(device)

        R, R_pinv = build_whitening(whitening, M, n, device, damp_alpha=damp_alpha)
        R_cache[p] = (R.cpu(), R_pinv.cpu())
        del M
        torch.cuda.empty_cache()

    print(f"\n[Apply Masked SVD] {model_name} | whitening={whitening}")
    for p in paths:
        fc = get_module(model, p)
        W = fc.weight.data
        bias = fc.bias.data.clone() if fc.bias is not None else None
        dtype, dev = fc.weight.dtype, fc.weight.device

        R, R_pinv = R_cache[p]
        R = R.to(dev, dtype=torch.float32)
        R_pinv = R_pinv.to(dev, dtype=torch.float32)

        W_tilde = W.float() @ R
        U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
        
        sqrt_sigma = torch.sqrt(S)
        W_u = U * sqrt_sigma
        W_v = (sqrt_sigma.unsqueeze(1) * Vt) @ R_pinv

        new_bias = bias

        if use_dobi:
            masked_fc = DobiSVDLinear(W_u, W_v, new_bias, dtype, dev, init_ratio=init_ratio)
        else:
            masked_fc = MaskedSVDLinear(W_u, W_v, new_bias, dtype, dev, init_ratio=init_ratio, path=p)
        replace_module(model, p, masked_fc)
        
        del R, R_pinv, W_u, W_v, masked_fc
        torch.cuda.empty_cache()

    return model

# ==========================================================================
# 6. Sliding Window Rank Search & Updates
# ==========================================================================
def get_all_blocks(model):
    """Forward-order chain of runnable sub-modules.

    ViT/DeiT: flat `model.blocks`. Swin: timm's SwinTransformerStage applies
    `downsample` (PatchMerging) BEFORE its blocks, and stage 0's downsample is an
    `Identity`. So per stage we emit the (non-Identity) downsample FIRST, then the
    blocks — matching the real forward order, so a sliding window that spans a
    stage boundary includes the PatchMerging in the correct position and the token
    dim (128->256->...) flows correctly. (The old version appended downsample last,
    feeding a 128-dim tensor into a 256-dim block -> LayerNorm shape crash.)
    """
    if hasattr(model, 'blocks') and len(getattr(model, 'blocks')) > 0:
        return list(model.blocks)
    blocks = []
    for layer in getattr(model, 'layers', []):
        ds = getattr(layer, 'downsample', None)
        if ds is not None and type(ds).__name__ != 'Identity':
            blocks.append(ds)
        if hasattr(layer, 'blocks'):
            blocks.extend(list(layer.blocks))
    return blocks

def sliding_window_rank_search(model_name, teacher_model, student_model, dataloader, dev, ratio, lambda_rank=1.0, lambda_lr=10.0, epochs=1, window_size=4, stride_size=1, lr=1.0, target_ratio_override=None, ce_last_windows=0, kd_temperature=2.0, stage2_decay=False, rel_mse=False, windows=None, stagger=2):
    print(f"\nStart Sliding Window Rank Search (Ratio = {ratio}, initial λ = {lambda_rank}, CE last windows = {ce_last_windows})...")
    
    # current_lambda is now managed per-window to prevent cross-window oscillation
    # lambda_lr is passed as an argument now
    
    teacher_blocks = get_all_blocks(teacher_model)
    student_blocks = get_all_blocks(student_model)
    
    for p in teacher_model.parameters():
        p.requires_grad = False
    for p in student_model.parameters():
        p.requires_grad = False
        
    all_masked_layers = []
    original_total_size = 0.0
    
    for name, module in student_model.named_modules():
        if isinstance(module, (MaskedSVDLinear, DobiSVDLinear)):
            all_masked_layers.append(module)
            in_dim = module.W_v.shape[1]
            out_dim = module.W_u.shape[0]
            original_total_size += (in_dim * out_dim)

    target_params = ratio * original_total_size
    full_svd_total_size = sum(m.W_u.shape[1] * (m.W_v.shape[1] + m.W_u.shape[0]) for m in all_masked_layers)
    sv_fraction = target_params / full_svd_total_size
    
    for module in all_masked_layers:
        r_max = module.W_u.shape[1]
        K = max(1, int(sv_fraction * r_max))
        if isinstance(module, DobiSVDLinear):
            module.k.fill_(float(K))
            
    num_blocks = len(teacher_blocks)
    # `windows` is a list of (start_block_idx, window_len) tuples. When None we fall
    # back to the classic fixed-size sliding schedule (behavior identical to before).
    # A custom schedule (e.g. stage-internal windows for Swin) is passed in verbatim.
    if windows is None:
        if num_blocks < window_size:
            print("Model too shallow for window size. Skipping rank search.")
            return
        windows = [(i, window_size) for i in range(0, num_blocks - window_size + 1, stride_size)]

    num_windows = len(windows)
    # The final `ce_last_windows` windows steer the mask via soft-KD cross-entropy
    # against the teacher's final logits instead of intermediate-feature MSE.
    ce_start_widx = num_windows - max(0, ce_last_windows)
    window_lambdas = [float(lambda_rank)] * num_windows
    window_target_ratios = []

    # Pre-calculate the starting parameter ratio for each window so Stage 2
    # strictly maintains the allocation decided by Stage 1!
    for w_idx, (i, w) in enumerate(windows):
        init_current_size = 0.0
        init_original_size = 0.0
        for b_idx in range(i, i + w):
            for name, module in student_blocks[b_idx].named_modules():
                if isinstance(module, (MaskedSVDLinear, DobiSVDLinear)):
                    with torch.no_grad():
                        if isinstance(module, MaskedSVDLinear):
                            # Use the hard boundary (Z > 0) to get the exact K intended by Stage 1,
                            # ignoring the artificial inflation of the soft Sigmoid tail!
                            kept_count = (module.Z > 0).sum().item()
                        else:
                            kept_count = int(module.k.item())
                            
                        in_dim = module.W_v.shape[1]
                        out_dim = module.W_u.shape[0]
                        init_current_size += kept_count * (in_dim + out_dim)
                        init_original_size += in_dim * out_dim
        if target_ratio_override is not None:
            window_target_ratios.append(target_ratio_override)
        else:
            window_target_ratios.append(init_current_size / init_original_size)
    
    for epoch in range(epochs):
        print(f"\n--- Rank Search Epoch {epoch+1}/{epochs} ---")
        # Staggered (even-odd) window schedule, ported from
        # svd_compression_stage2_even_odd.py: each epoch processes every OTHER
        # window, alternating the parity (offset = epoch % 2) so consecutive
        # epochs together cover all windows. Only the Stage-2 rank search is
        # staggered; Stage-3 fine-tuning keeps its full per-epoch sweep. Indexing
        # by window position keeps this valid for both the classic fixed schedule
        # and any custom `windows` list.
        start_offset = epoch % max(1, stagger)
        for w_idx in range(start_offset, len(windows), max(1, stagger)):
            i, w = windows[w_idx]
            use_ce = ce_last_windows > 0 and w_idx >= ce_start_widx
            loss_type = "KD-CE" if use_ce else "MSE"
            print(f"  Window {w_idx}/{num_windows-1} (blocks {i} to {i+w-1}) [{loss_type}]...")

            current_lambda = window_lambdas[w_idx]
            target_window_ratio = window_target_ratios[w_idx]

            teacher_targets = []
            teacher_logits = []
            student_inputs = []

            def teacher_hook(module, inp, out):
                teacher_targets.append(out.detach().cpu())

            def student_hook(module, args):
                student_inputs.append([a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args])

            # MSE windows cache the intermediate teacher feature; CE windows instead
            # cache the teacher's final logits (captured from the forward return).
            h_t = None
            if not use_ce:
                h_t = teacher_blocks[i + w - 1].register_forward_hook(teacher_hook)
            h_s = student_blocks[i].register_forward_pre_hook(student_hook)

            with torch.no_grad():
                for imgs, _ in tqdm(dataloader, desc=f"Caching Window {i}", leave=False):
                    imgs = imgs.to(dev)
                    t_out = teacher_model(imgs)
                    if use_ce:
                        teacher_logits.append(t_out.detach().cpu())
                    student_model(imgs)

            if h_t is not None:
                h_t.remove()
            h_s.remove()
            
            params_to_opt = []
            for b_idx in range(i, i + w):
                student_blocks[b_idx].train()
                for name, module in student_blocks[b_idx].named_modules():
                    if isinstance(module, MaskedSVDLinear):
                        module.Z.requires_grad = True
                        params_to_opt.append(module.Z)
                    elif isinstance(module, DobiSVDLinear):
                        module.k.requires_grad = True
                        params_to_opt.append(module.k)
                        
            optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=0.0)
            criterion = torch.nn.MSELoss()
            
            total_loss = 0.0
            total_mse = 0.0
            total_size_loss = 0.0
            for batch_idx in tqdm(range(len(student_inputs)), desc=f"Window {i//stride_size}", leave=False):
                optimizer.zero_grad()
                
                args = [a.to(dev) if isinstance(a, torch.Tensor) else a for a in student_inputs[batch_idx]]

                x = student_blocks[i](*args)
                for b_idx in range(i + 1, i + w):
                    x = student_blocks[b_idx](x)

                if use_ce:
                    # Run the remaining (frozen-mask) student blocks + head to logits.
                    for b_idx in range(i + w, num_blocks):
                        x = student_blocks[b_idx](x)
                    logits = student_model.forward_head(student_model.norm(x))
                    target_logits = teacher_logits[batch_idx].to(dev)
                    mse_loss = soft_kd_loss(logits, target_logits, T=kd_temperature)
                else:
                    target = teacher_targets[batch_idx].to(dev)
                    mse_loss = feature_mse_loss(x, target, rel_mse=rel_mse)

                # Local Window Size Penalty
                current_size = 0.0
                window_original_size = 0.0
                for b_idx in range(i, i + w):
                    for name, module in student_blocks[b_idx].named_modules():
                        if isinstance(module, (MaskedSVDLinear, DobiSVDLinear)):
                            mask = module.get_mask()
                            in_dim = module.W_v.shape[1]
                            out_dim = module.W_u.shape[0]
                            current_size = current_size + mask.sum() * (in_dim + out_dim)
                            window_original_size += in_dim * out_dim
                    
                size_ratio = current_size / window_original_size
                size_loss = current_lambda * torch.abs(size_ratio - target_window_ratio)
                
                loss = mse_loss + size_loss
                loss.backward()
                optimizer.step()
                
                # Dynamically update lambda to aggressively enforce target ratio for both stages
                current_lambda = max(0.0, current_lambda + lambda_lr * (size_ratio.item() - target_window_ratio))
                window_lambdas[w_idx] = current_lambda
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_size_loss += size_loss.item()
                
            n_batches = max(1, len(student_inputs))
            window_maintained = 0
            window_total = 0
            for _b in range(i, i + w):
                for name, module in student_blocks[_b].named_modules():
                    if isinstance(module, MaskedSVDLinear):
                        window_maintained += (module.Z > 0).sum().item()
                        window_total += module.Z.shape[0]
                    elif isinstance(module, DobiSVDLinear):
                        window_maintained += int(module.k.item())
                        window_total += module.W_u.shape[1]
            print(f"    Loss: {total_loss/n_batches:.6f} ({loss_type}: {total_mse/n_batches:.6f}, Size: {total_size_loss/n_batches:.6f}, λ: {current_lambda:.2f}) | SVs kept in window: {window_maintained}/{window_total}")
                
            for b_idx in range(i, i + w):
                student_blocks[b_idx].eval()
                for name, module in student_blocks[b_idx].named_modules():
                    if isinstance(module, MaskedSVDLinear):
                        module.Z.requires_grad = False
            
            del teacher_targets, teacher_logits, student_inputs, optimizer
            torch.cuda.empty_cache()

        # Anneal Temperature for Stage 2 (was missing in sliding window search!)
        if stage2_decay:
            for module in all_masked_layers:
                if isinstance(module, MaskedSVDLinear):
                    decay_factor = (0.1 / 10.0) ** (1.0 / epochs)
                    module.temperature = max(0.1, module.temperature * decay_factor)

def global_rank_search(model_name, teacher_model, student_model, dataloader, dev, ratio, lambda_rank=1.0, lambda_lr=10.0, epochs=1, lr=1e-2, stage2_decay=False):
    print(f"\nStart Global Rank Search (Ratio = {ratio}, initial λ = {lambda_rank})...")
    
    all_masked_layers = []
    original_total_size = 0.0
    
    for name, module in student_model.named_modules():
        if isinstance(module, (MaskedSVDLinear, DobiSVDLinear)):
            all_masked_layers.append(module)
            in_dim = module.W_v.shape[1]
            out_dim = module.W_u.shape[0]
            original_total_size += (in_dim * out_dim)

    current_lambda = float(lambda_rank)

    target_params = ratio * original_total_size
    
    params_to_opt = []
    student_model.train()
    for module in all_masked_layers:
        if isinstance(module, MaskedSVDLinear):
            module.Z.requires_grad = True
            params_to_opt.append(module.Z)
        elif isinstance(module, DobiSVDLinear):
            module.k.requires_grad = True
            params_to_opt.append(module.k)
            
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=0.0)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        print(f"\n--- Rank Search Epoch {epoch+1}/{epochs} ---")
        
        total_loss = 0.0
        total_mse = 0.0
        total_size_loss = 0.0
        n_batches = 0
        
        for imgs, _ in tqdm(dataloader, desc="Global Rank Search", leave=False):
            imgs = imgs.to(dev)
            
            with torch.no_grad():
                teacher_out = teacher_model(imgs)
                
            student_out = student_model(imgs)
            
            mse_loss = criterion(student_out, teacher_out)
            
            # Global Size Penalty
            current_size = 0.0
            for module in all_masked_layers:
                mask = module.get_mask()
                req_grad = getattr(module, 'Z', getattr(module, 'k', None)).requires_grad
                if not req_grad:
                    mask = mask.detach()
                in_dim = module.W_v.shape[1]
                out_dim = module.W_u.shape[0]
                current_size = current_size + mask.sum() * (in_dim + out_dim)
                
            size_ratio = current_size / original_total_size
            size_loss = current_lambda * torch.abs(size_ratio - ratio)
            
            loss = mse_loss + size_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Dynamically update lambda to aggressively enforce target ratio for both stages
            current_lambda = max(0.0, current_lambda + lambda_lr * (size_ratio.item() - ratio))
                
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_size_loss += size_loss.item()
            n_batches += 1
            
        n_batches = max(1, n_batches)
        
        window_maintained = 0
        window_total = 0
        
        # Anneal Temperature for Stage 2
        if stage2_decay:
            for module in all_masked_layers:
                if isinstance(module, MaskedSVDLinear):
                    decay_factor = (0.1 / 10.0) ** (1.0 / epochs)
                    module.temperature = max(0.1, module.temperature * decay_factor)
                
        # Get one layer's temperature to print
        current_temp = None
        for module in all_masked_layers:
            if isinstance(module, MaskedSVDLinear):
                current_temp = module.temperature
                break
        temp_str = f", Temp: {current_temp:.2f}" if current_temp is not None else ""
        
        total_kept = 0
        full_svd_total_size = 0
        for module in all_masked_layers:
            if isinstance(module, MaskedSVDLinear):
                total_kept += (torch.sigmoid(module.Z / module.temperature) > 0.5).sum().item()
                full_svd_total_size += module.Z.shape[0]
            elif isinstance(module, DobiSVDLinear):
                total_kept += int(module.k.item())
                full_svd_total_size += module.W_u.shape[1]
                
        avg_loss = total_loss/n_batches
        avg_mse = total_mse/n_batches
        avg_size_penalty = total_size_loss/n_batches
        print(f"    Loss: {avg_loss:.6f} (MSE: {avg_mse:.6f}, Size: {avg_size_penalty:.6f}, λ: {current_lambda:.2f}{temp_str}) | SVs kept globally: {total_kept}/{full_svd_total_size}")
        
    student_model.eval()
    for module in all_masked_layers:
        if isinstance(module, MaskedSVDLinear):
            module.Z.requires_grad = False
        elif isinstance(module, DobiSVDLinear):
            module.k.requires_grad = False
    
    torch.cuda.empty_cache()

def assign_ranks_and_slice(model_name, model, out_dir="results", use_cayley=False, stages_str=""):
    print("\nAssigning ranks and slicing chosen singular values...")
    paths = list_fc_paths(model, model_name)
    
    layer_ranks = []
    
    import matplotlib.pyplot as plt
    import os
    import math
    
    num_paths = len(paths)
    cols = 4
    rows = math.ceil(num_paths / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if num_paths == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    plot_idx = 0
    
    for p in paths:
        masked_fc = get_module(model, p)
        
        if isinstance(masked_fc, MaskedSVDLinear):
            Z = masked_fc.Z.detach()
            
            # Plot all layers
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                mask_vals = torch.sigmoid(Z / masked_fc.temperature).cpu().numpy()
                ax.plot(mask_vals, '.', label='Mask Value (Sigmoid)', color='blue', alpha=0.5)
                ax.axhline(0.5, color='red', linestyle='--', label='Keep Threshold (>0.5)')
                ax.set_title(f'Learned Mask for {p}')
                ax.set_xlabel('Singular Value Index (0 = Largest SV)')
                ax.set_ylabel('Mask Gate Value')
                ax.legend()
                plot_idx += 1
                
            active_indices = torch.nonzero(Z > 0).squeeze(-1)
        elif isinstance(masked_fc, DobiSVDLinear):
            k_val = int(round(masked_fc.k.item()))
            k_val = max(1, min(k_val, masked_fc.W_u.shape[1]))
            active_indices = torch.arange(k_val, device=masked_fc.W_u.device)
            
        W_u_full = masked_fc.W_u.data
        W_v_full = masked_fc.W_v.data
        bias = masked_fc.bias.data if masked_fc.bias is not None else None
        dtype, dev = W_u_full.dtype, W_u_full.device
        
        if len(active_indices) == 0:
            active_indices = torch.tensor([0], device=dev)
            
        layer_ranks.append(str(len(active_indices)))
            
        W_u_new = W_u_full[:, active_indices]
        W_v_new = W_v_full[active_indices, :]
        
        new_fc = make_factored_linear(W_u_new, W_v_new, bias, dtype, dev, use_cayley=use_cayley)
        replace_module(model, p, new_fc)
        
    if plot_idx > 0:
        plt.tight_layout()
        suffix = f"_{stages_str}" if stages_str else ""
        plot_path = os.path.join(out_dir, f"{model_name_for_filename(model_name)}{suffix}_learned_masks.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"\n[!] Saved Mask Plot to {plot_path} so you can visually see the scattered Singular Values!")
        
    layer_ranks_str = ",".join(layer_ranks)
    return model, layer_ranks_str

def soft_kd_loss(student_logits, teacher_logits, T=2.0):
    """Soft cross-entropy (KL) between the student and (frozen) teacher logits.

    Standard Hinton distillation: match the temperature-softened teacher
    distribution, rescaled by T^2 so the gradient magnitude is T-invariant.
    """
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)

def feature_mse_loss(x, target, rel_mse=False, eps=1e-8):
    """Feature reconstruction loss for the MSE windows.

    Default = plain MSE (unchanged behavior). With rel_mse=True, normalize by the
    teacher feature energy: mean((x-t)^2) / mean(t^2). Since mean(t^2) is constant
    w.r.t. the student params, this is a per-window rescaling of the MSE gradient,
    making the reconstruction-vs-size-penalty balance consistent across windows even
    when deep-block activation magnitudes explode (massive activations).
    """
    mse = F.mse_loss(x, target)
    if rel_mse:
        return mse / (target.pow(2).mean() + eps)
    return mse

def sliding_window_update(model_name, teacher_model, student_model, dataloader, dev, epochs=5, window_size=4, stride_size=1, lr=1e-4, freeze_s=False, ce_last_windows=0, kd_temperature=2.0, rel_mse=False, windows=None):
    print(f"\nStart Sliding Window Fine-Tuning (Window Size = {window_size}, CE last windows = {ce_last_windows})...")

    teacher_blocks = get_all_blocks(teacher_model)
    student_blocks = get_all_blocks(student_model)

    for p in teacher_model.parameters():
        p.requires_grad = False

    for p in student_model.parameters():
        p.requires_grad = False

    num_blocks = len(teacher_blocks)
    # `windows` = list of (start_block_idx, window_len). None -> classic fixed-size
    # sliding schedule (identical to before). A custom schedule (e.g. Swin
    # stage-internal windows) is used verbatim.
    if windows is None:
        if num_blocks < window_size:
            print("Model too shallow for window size. Skipping.")
            return 0.0
        windows = [(i, window_size) for i in range(0, num_blocks - window_size + 1, stride_size)]

    if epochs == 0:
        print("Epochs = 0, skipping fine-tuning entirely.")
        return 0.0

    num_windows = len(windows)
    # The final `ce_last_windows` windows use soft-KD cross-entropy against the
    # teacher's final logits instead of intermediate-feature MSE.
    ce_start_widx = num_windows - max(0, ce_last_windows)

    avg_losses = []

    for w_idx, (i, w) in enumerate(windows):
        use_ce = ce_last_windows > 0 and w_idx >= ce_start_widx
        loss_type = "KD-CE" if use_ce else "MSE"
        print(f"\n  Optimizing Window {w_idx}/{num_windows-1} (blocks {i} to {i+w-1}) [{loss_type}]...")

        teacher_targets = []
        teacher_logits = []
        student_inputs = []

        def teacher_hook(module, inp, out):
            teacher_targets.append(out.detach().cpu())

        def student_hook(module, args):
            student_inputs.append([a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in args])

        # MSE windows cache the intermediate teacher feature; CE windows instead
        # cache the teacher's final logits (captured from the forward return).
        h_t = None
        if not use_ce:
            h_t = teacher_blocks[i + w - 1].register_forward_hook(teacher_hook)
        h_s = student_blocks[i].register_forward_pre_hook(student_hook)

        with torch.no_grad():
            for imgs, _ in tqdm(dataloader, desc=f"Caching Window {i}", leave=False):
                imgs = imgs.to(dev)
                t_out = teacher_model(imgs)
                if use_ce:
                    teacher_logits.append(t_out.detach().cpu())
                student_model(imgs)

        if h_t is not None:
            h_t.remove()
        h_s.remove()

        params_to_opt = []
        for b_idx in range(i, i + w):
            student_blocks[b_idx].train()
            for name, p in student_blocks[b_idx].named_parameters():
                if freeze_s and 'fc_s' in name:
                    continue
                if ('.0.weight' in name or '.1.weight' in name or 'fc_v' in name or 'fc_s' in name or 'fc_u' in name):
                    p.requires_grad = True
                    params_to_opt.append(p)

        optimizer = torch.optim.AdamW(params_to_opt, lr=lr)
        criterion = torch.nn.MSELoss()

        total_loss = 0.0
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx in tqdm(range(len(student_inputs)), desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                optimizer.zero_grad()

                args = [a.to(dev) if isinstance(a, torch.Tensor) else a for a in student_inputs[batch_idx]]

                x = student_blocks[i](*args)
                for b_idx in range(i + 1, i + w):
                    x = student_blocks[b_idx](x)

                if use_ce:
                    # Run the remaining (frozen) student blocks + head to logits.
                    for b_idx in range(i + w, num_blocks):
                        x = student_blocks[b_idx](x)
                    logits = student_model.forward_head(student_model.norm(x))
                    target_logits = teacher_logits[batch_idx].to(dev)
                    loss = soft_kd_loss(logits, target_logits, T=kd_temperature)
                else:
                    target = teacher_targets[batch_idx].to(dev)
                    loss = feature_mse_loss(x, target, rel_mse=rel_mse)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"    Epoch {epoch+1}/{epochs} | {loss_type} Loss: {total_loss/max(1, len(student_inputs)):.6f}")

        avg_losses.append(total_loss / max(1, len(student_inputs)))

        for b_idx in range(i, i + w):
            student_blocks[b_idx].eval()
            for name, p in student_blocks[b_idx].named_parameters():
                p.requires_grad = False

        del teacher_targets, teacher_logits, student_inputs, optimizer
        torch.cuda.empty_cache()

    return sum(avg_losses) / max(1, len(avg_losses))

# ==========================================================================
# 7. Evaluation
# ==========================================================================
@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Eval"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return correct / total * 100.0

def append_csv(filepath, rows):
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Model_Name", "Method", "Compression_Ratio", "Accuracy", "MACs_GB", "Parameter_Count_M", "Avg_Window_MSE_Loss", "Maintained_SVs_Per_Layer"])
        for row in rows:
            writer.writerow(row)

def eval_vit_sliding_window_train_rank(models, ratios, out_dir, whitening="gptq_damp", window_size=4, rank_search_window_size=None, stride_size=1, 
                                       epochs=2, rank_search_epochs=1, lr=1e-4, rank_lr=1.0, lambda_rank=1.0, lambda_lr=10.0,
                                       calib_size=8192, use_dobi=False, run_global_search=False,
                                       run_ipca_update=False, stage2_end_to_end=False, use_cayley=False, freeze_s=False, skip_rank_search=False, dobi_args=None,
                                       ce_last_windows_list=None, kd_temperature=2.0, ce_rank_search=False, stage2_decay=False,
                                       stage2_temperature=None, rel_mse=False, window_builder=None, stage2_stagger=2):
    if ce_last_windows_list is None:
        ce_last_windows_list = [0]
    os.makedirs(out_dir, exist_ok=True)
    rs_window_size = rank_search_window_size if rank_search_window_size is not None else window_size
    AttentionClass = timm.models.vision_transformer.Attention
    custom_ops = {torch.nn.GELU: count_gelu, torch.nn.LayerNorm: count_layernorm,
                  torch.nn.Softmax: count_softmax, AttentionClass: count_attention}
    # Swin uses WindowAttention (not vision_transformer.Attention); its forward
    # input is (num_windows*B, window_area, C) with a .num_heads attr, so the same
    # count_attention formula applies and keeps MACs accounting consistent.
    try:
        from timm.models.swin_transformer import WindowAttention
        custom_ops[WindowAttention] = count_attention
    except Exception:
        pass
    input_tensor = torch.randn(1, 3, 224, 224).to(device)

    for model_name in models:
        safe = model_name_for_filename(model_name)
        print(f"\n{'='*60}\n{model_name}  (Sliding Window Rank Search | {whitening})\n{'='*60}")

        _, test_loader, _, _, cali_loader = get_timm_dataloaders(
            batch_size=BATCH_SIZE, model_name=model_name,
            calibration_size=calib_size, calibration_seed=CALIBRATION_SEED)

        # Both MaskedSVDLinear and DobiSVDLinear use whitening statistics in our custom pipeline.
        print("Calibrating statistics...")
        stats_cache = calibrate_stats_vit(model_name, cali_loader, device)

        for ratio in ratios:
            stages = []
            if not skip_rank_search:
                if args.two_stage_rank_search:
                    stages.append("s1")
                    stage2_eps = args.stage2_epochs if args.stage2_epochs is not None else rank_search_epochs
                    if stage2_eps > 0:
                        stages.append("s2")
                else:
                    stages.append("s2")
            if epochs > 0:
                stages.append("s3")
            stages_str = "-".join(stages) if stages else "uniform"

            folder_suffix = "_cayley" if use_cayley else ""
            if skip_rank_search:
                ratio_dir = os.path.join(out_dir, f"results_{safe}_{ratio}_uniform{folder_suffix}")
            else:
                ratio_dir = os.path.join(out_dir, f"results_{safe}_{ratio}{folder_suffix}")
            os.makedirs(ratio_dir, exist_ok=True)
            out_csv = os.path.join(ratio_dir, f"{stages_str}_sliding_window_train_rank_w{window_size}_s{stride_size}_e{epochs}_rs{rank_search_epochs}_c{calib_size}.csv")
            
            print(f"\n--- Ratio {ratio} ---")
            teacher_model = timm.create_model(model_name, pretrained=True).to(device)
            teacher_model.eval()

            # Optional custom window schedule (e.g. Swin stage-internal windows). None
            # -> the classic fixed-size sliding schedule inside the sliding functions.
            rs_windows = window_builder(teacher_model, rs_window_size, stride_size) if window_builder is not None else None
            ft_windows = window_builder(teacher_model, window_size, stride_size) if window_builder is not None else None

            def build_sliced_student(rs_ce):
                """Run rank search (Stages 1-3) and return (sliced_student, layer_ranks_str).

                rs_ce = number of FINAL windows that use soft-KD cross-entropy (vs the
                teacher's logits) instead of MSE during Stage-2 sliding-window rank search.
                """
                student_model = timm.create_model(model_name, pretrained=True).to(device)
                student_model.eval()

                if run_global_search and use_dobi:
                    print("\n[!] Original Dobi-SVD Rank Allocation (Activation SVD)...")
                    gamma_dict = dobisvd.train_gamma(model_name, ratio, cali_loader, dobi_args)

                    # Convert gamma to absolute rank integer
                    rank_of = {}
                    for path, gamma_val in gamma_dict.items():
                        fc = get_module(teacher_model, path)
                        r = int(max(1, min(fc.in_features, fc.out_features, round(gamma_val))))
                        rank_of[path] = r

                    if run_ipca_update:
                        print("\n[!] Using IPCA Weight Update instead of Whitening...")
                        new_weights, rank_of = dobisvd.ipca_weight_update(model_name, gamma_dict, dobi_args.beta, cali_loader, dobi_args)
                        student_model = dobisvd.build_compressed_model(model_name, gamma_dict, new_weights, rank_of)
                        student_model.eval()
                        layer_ranks_str = ",".join(str(rank_of[p]) for p in rank_of)
                        del new_weights
                        torch.cuda.empty_cache(); gc.collect()
                    else:
                        # Initialize MaskedSVDLinear WITHOUT DobiSVDLinear continuous slider
                        student_model = apply_masked_svd(
                            model_name=model_name,
                            model=student_model,
                            stats_cache=stats_cache,
                            device=device,
                            whitening=whitening,
                            use_dobi=False,
                            init_ratio=ratio
                        )

                        # Hardcode the Z parameters to cleanly slice at the desired rank
                        for p in rank_of:
                            module = get_module(student_model, p)
                            if isinstance(module, MaskedSVDLinear):
                                K = rank_of[p]
                                with torch.no_grad():
                                    module.Z.fill_(-3.0)
                                    module.Z[:K] = 3.0

                        # Skip rank search, jump to assign_ranks_and_slice
                        student_model, layer_ranks_str = assign_ranks_and_slice(model_name, student_model, out_dir=ratio_dir, use_cayley=use_cayley, stages_str=stages_str)
                elif args.two_stage_rank_search:
                    print("\n" + "="*50)
                    print(">>> STAGE 1: Layer Rank Search (DobiSVD k parameter)")
                    print("="*50)
                    student_model = apply_masked_svd(
                        model_name=model_name,
                        model=student_model,
                        stats_cache=stats_cache,
                        device=device,
                        whitening=whitening,
                        use_dobi=True,
                        init_ratio=ratio
                    )
                    global_rank_search(model_name, teacher_model, student_model, cali_loader, device,
                                       ratio=ratio, lambda_rank=lambda_rank, lambda_lr=lambda_lr,
                                       epochs=rank_search_epochs, lr=rank_lr, stage2_decay=stage2_decay)

                    # Extract learned k values
                    rank_of = {}
                    for name, module in student_model.named_modules():
                        if isinstance(module, DobiSVDLinear):
                            r_max = module.W_u.shape[1]
                            k_val = int(round(module.k.item()))
                            k_val = max(1, min(k_val, r_max))
                            # name corresponds to path since we named it cleanly, wait, name in named_modules might be slightly different than paths
                            # But actually, DobiSVDLinear is injected at the path. So `name` is the path!
                            rank_of[name] = k_val

                    print(f"\nExtracted Ranks from Stage 1: {rank_of}")

                    print("\n" + "="*50)
                    print(">>> STAGE 2: Individual SV Rank Search (MaskedSVD Z_i)")
                    print("="*50)

                    if args.stage2_epochs == 0:
                        print("\n[!] Ablation Mode: Skipping Stage 2 completely. Slicing directly from Stage 1 (DobiSVD)...")
                        student_model, layer_ranks_str = assign_ranks_and_slice(model_name, student_model, out_dir=ratio_dir, use_cayley=use_cayley, stages_str=stages_str)
                        return student_model, layer_ranks_str

                    # Re-initialize the model to wipe DobiSVDLinear and insert MaskedSVDLinear
                    del student_model
                    torch.cuda.empty_cache(); gc.collect()
                    student_model = timm.create_model(model_name, pretrained=True).to(device)
                    student_model.eval()

                    student_model = apply_masked_svd(
                        model_name=model_name,
                        model=student_model,
                        stats_cache=stats_cache,
                        device=device,
                        whitening=whitening,
                        use_dobi=False,
                        init_ratio=rank_of  # Pass the dictionary here!
                    )

                    # Optional override of the Stage-2 sigmoid-gate temperature (default 10.0,
                    # set in MaskedSVDLinear.__init__). Only for ablating T; None = untouched.
                    if stage2_temperature is not None:
                        for m in student_model.modules():
                            if isinstance(m, MaskedSVDLinear):
                                m.temperature = float(stage2_temperature)

                    # Run Stage 2 search on Z_i parameters
                    stage2_epochs = args.stage2_epochs if args.stage2_epochs is not None else args.rank_search_epochs
                    if stage2_end_to_end:
                        print(f"\nStart End-to-End Stage 2 Rank Search (Ratio = {ratio}, initial λ = {lambda_rank})...")
                        global_rank_search(model_name, teacher_model, student_model, cali_loader, device,
                                           ratio=ratio, lambda_rank=lambda_rank, lambda_lr=lambda_lr,
                                           epochs=stage2_epochs, lr=rank_lr, stage2_decay=stage2_decay)
                    else:
                        sliding_window_rank_search(model_name, teacher_model, student_model, cali_loader, device,
                                                   ratio=ratio, lambda_rank=lambda_rank, lambda_lr=lambda_lr,
                                                   epochs=stage2_epochs, window_size=rs_window_size,
                                                   stride_size=stride_size, lr=rank_lr,
                                                   ce_last_windows=rs_ce, kd_temperature=kd_temperature, stage2_decay=stage2_decay, rel_mse=rel_mse, windows=rs_windows)

                    # STAGE 3: Finalize Masked SVD
                    student_model, layer_ranks_str = assign_ranks_and_slice(model_name, student_model, out_dir=ratio_dir, use_cayley=use_cayley, stages_str=stages_str)

                else:
                    # We now ALWAYS use our custom rank search pipeline (global or sliding window).
                    # If use_dobi=True, it will use DobiSVDLinear (continuous parameter).

                    # 2. Rank Search
                    if skip_rank_search:
                        print(f"\nSkipping Dynamic Rank Search. Applying pure Uniform Rank (Ratio = {ratio})...")
                        student_model = apply_v2_method(
                            model_name=model_name,
                            model=student_model,
                            stats_cache=stats_cache,
                            ratio=float(ratio),
                            device=device,
                            whitening=whitening,
                            use_cayley=use_cayley
                        )
                        # Extract the assigned uniform ranks for logging
                        paths = list_fc_paths(student_model, model_name)
                        layer_ranks = []
                        for p in paths:
                            fc = get_module(student_model, p)
                            if isinstance(fc, nn.Sequential) and hasattr(fc, 'fc_v'):
                                layer_ranks.append(str(fc.fc_v.out_features))
                            else:
                                # Fallback if not a factored sequential
                                layer_ranks.append(str(fc.out_features))
                        layer_ranks_str = ",".join(layer_ranks)

                    else:
                        # 1. Apply Masked SVD
                        student_model = apply_masked_svd(
                            model_name=model_name,
                            model=student_model,
                            stats_cache=stats_cache,
                            device=device,
                            whitening=whitening,
                            use_dobi=use_dobi,
                            init_ratio=ratio
                        )

                        # Apply Stage-2 sigmoid-gate temperature override for custom MaskedSVD pipeline
                        if stage2_temperature is not None:
                            for m in student_model.modules():
                                if isinstance(m, MaskedSVDLinear):
                                    m.temperature = float(stage2_temperature)

                        if run_global_search:
                            print(f"\nStart Global Rank Search (Ratio = {ratio}, initial λ = {lambda_rank})...")
                            global_rank_search(model_name, teacher_model, student_model, cali_loader, device,
                                               ratio=ratio, lambda_rank=lambda_rank, lambda_lr=lambda_lr,
                                               epochs=rank_search_epochs, lr=rank_lr, stage2_decay=stage2_decay)
                        
                        if args.two_stage_rank_search or not run_global_search:
                            print(f"\nStart Sliding Window Rank Search (Window Size = {rs_window_size}, Ratio = {ratio}, initial λ = {lambda_rank})...")
                            sliding_window_rank_search(model_name, teacher_model, student_model, cali_loader, device,
                                                       ratio=ratio, lambda_rank=lambda_rank, lambda_lr=lambda_lr,
                                                       epochs=args.stage2_epochs if args.stage2_epochs is not None else rank_search_epochs, 
                                                       window_size=rs_window_size,
                                                       stride_size=stride_size, lr=rank_lr, target_ratio_override=ratio,
                                                       ce_last_windows=rs_ce, kd_temperature=kd_temperature, stage2_decay=stage2_decay, rel_mse=rel_mse, windows=rs_windows, stagger=stage2_stagger)

                        # 3. Finalize Masked SVD
                        student_model, layer_ranks_str = assign_ranks_and_slice(model_name, student_model, out_dir=ratio_dir, use_cayley=use_cayley, stages_str=stages_str)

                return student_model, layer_ranks_str

            # 4. Sliding Window Fine-Tuning
            # Sweep the number of final windows fine-tuned with soft-KD cross-entropy.
            # When --ce_rank_search is set, the CE count ALSO drives Stage-2 rank
            # allocation, so the sliced student is rebuilt per sweep value; otherwise
            # rank search is run once and deep-copied across the sweep.
            sliced_cache = None
            sliced_ranks_cache = None
            for ce_lw in ce_last_windows_list:
                if ce_rank_search or sliced_cache is None:
                    if sliced_cache is not None:
                        del sliced_cache
                        torch.cuda.empty_cache(); gc.collect()
                    rs_ce = ce_lw if ce_rank_search else 0
                    sliced_cache, sliced_ranks_cache = build_sliced_student(rs_ce)
                layer_ranks_str = sliced_ranks_cache
                student_ft = copy.deepcopy(sliced_cache)

                avg_mse = sliding_window_update(
                    model_name=model_name,
                    teacher_model=teacher_model,
                    student_model=student_ft,
                    dataloader=cali_loader,
                    dev=device,
                    epochs=epochs,
                    window_size=window_size,
                    stride_size=stride_size,
                    lr=lr,
                    freeze_s=freeze_s,
                    ce_last_windows=ce_lw,
                    kd_temperature=kd_temperature,
                    rel_mse=rel_mse,
                    windows=ft_windows,
                )

                acc = evaluate_acc(student_ft, test_loader, device)

                macs, params = eval_macs(model_name, student_ft, input_tensor, custom_ops)
                macs_gb = macs / 1e9
                params_m = params / 1e6

                if skip_rank_search:
                    method_str = f"uniform_rank_{whitening}_r={ratio}"
                elif use_dobi:
                    method_str = f"sliding_window_dobisvd_r={ratio}"
                elif stage2_end_to_end:
                    method_str = f"end_to_end_train_rank_{whitening}_r={ratio}"
                else:
                    method_str = f"sliding_window_train_rank_{whitening}_r={ratio}"

                if use_cayley:
                    method_str += "_cayley"
                # Tag the CE window count; append "rs" when it also drove rank search.
                method_str += f"_ce{ce_lw}" + ("rs" if ce_rank_search else "")
                # Ablation knobs: only tagged when explicitly overridden, so default
                # runs keep exactly the method_str they had before.
                if stage2_temperature is not None:
                    method_str += f"_T{stage2_temperature:g}"
                if rank_search_window_size is not None:
                    method_str += f"_rsw{rs_window_size}"
                if rel_mse:
                    method_str += "_relmse"
                if window_builder is not None:
                    method_str += "_stageint"

                print(f"Train Rank Acc (ratio={ratio}, ce_last_windows={ce_lw}, ce_rank_search={ce_rank_search}, T={kd_temperature}): {acc:.2f}% | MACs: {macs_gb:.2f}G | Params: {params_m:.2f}M | Avg Loss: {avg_mse:.6f} | SVs per layer: {layer_ranks_str}")
                row = [model_name, method_str, f"{ratio:.6f}", f"{acc:.6f}", f"{macs_gb:.6f}", f"{params_m:.6f}", f"{avg_mse:.6f}", layer_ranks_str]
                append_csv(out_csv, [row])
                print(f"Saved: {out_csv}")

                del student_ft
                torch.cuda.empty_cache(); gc.collect()

            del teacher_model, sliced_cache
            torch.cuda.empty_cache(); gc.collect()

        del test_loader, cali_loader, stats_cache
        torch.cuda.empty_cache(); gc.collect()

# ==========================================================================
# 8. Main (CLI)
# ==========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filter", type=str, default="")
    parser.add_argument("--ratio", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="Comma-separated list of ratios, e.g., '0.9,0.8,0.7'")
    parser.add_argument("--out_dir", type=str, default="results/ourmethod_test")
    parser.add_argument("--run_sliding_window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--whitening", type=str, default="gptq_damp")
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--rank_search_window_size", type=int, default=None, help="If set, overrides window_size during the rank search phase.")
    parser.add_argument("--stride_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=4, help="Stage-3 sliding-window fine-tuning epochs")
    parser.add_argument("--rank_search_epochs", type=int, default=20, help="Stage-1 global rank-search epochs")
    parser.add_argument("--stage2_epochs", type=int, default=30, help="Stage-2 staggered sliding-window rank-search epochs")
    parser.add_argument("--calib_size", type=int, default=1024)
    parser.add_argument("--lambda_rank", type=float, default=1.0)
    parser.add_argument("--lambda_lr", type=float, default=10.0, help="Learning rate for Dual Ascent lambda.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank_lr", type=float, default=5.0, help="Learning rate for the rank parameter k during search.")
    parser.add_argument("--dobi_svd_sliding", action="store_true",
                        help="use the real Dobi-SVD rank allocation (activation truncation + IPCA, as in dobisvd.py)")
    parser.add_argument("--two_stage_rank_search", action=argparse.BooleanOptionalAction, default=True, help="Stage 1: DobiSVD k search. Stage 2: MaskedSVD individual Z search.")
    parser.add_argument("--run_global_search", action="store_true")
    parser.add_argument("--run_ipca_update", action="store_true", help="Use original Dobi-SVD IPCA weight update instead of Whitening")
    parser.add_argument("--stage2_end_to_end", action="store_true", help="Run Stage 2 end-to-end instead of sliding window.")
    parser.add_argument("--skip_rank_search", action="store_true", help="Skip dynamic rank search and use uniform rank allocation.")
    parser.add_argument("--cayley", action="store_true", help="Use Cayley Orthogonal Fine-Tuning for Stage 4")
    parser.add_argument("--freeze_s", action="store_true", help="Freeze the singular value scaling factors during Stage 4 fine-tuning (only applies if --cayley is used)")
    parser.add_argument("--ce_last_windows", type=str, default="0",
                        help="Comma-separated sweep of how many FINAL Stage-4 sliding windows to fine-tune with soft-KD cross-entropy (vs teacher logits) instead of MSE. e.g. '0,1,2,3'. Rank search is run once and reused across the sweep.")
    parser.add_argument("--kd_temperature", type=float, default=2.0,
                        help="Softmax temperature for the soft-KD cross-entropy loss on the last windows.")
    parser.add_argument("--ce_rank_search", action="store_true",
                        help="Also apply the soft-KD cross-entropy loss to the last N windows of the Stage-2 sliding-window RANK SEARCH (using the same swept --ce_last_windows count). This makes rank allocation depend on the sweep value, so rank search is rebuilt per sweep value instead of reused.")
    parser.add_argument("--stage2_decay", action="store_true", help="Decay temperature to 0.1 instead of keeping it fixed at 10.0")
    parser.add_argument("--stage2_temperature", type=float, default=10.0,
                        help="Stage-2 sigmoid-gate temperature (fixed, no decay). Requested setting: 10.")
    parser.add_argument("--rel_mse", action="store_true",
                        help="Use relative (energy-normalized) MSE for the sliding-window MSE loss in rank search & fine-tuning: mean((x-t)^2)/mean(t^2). Makes the loss scale consistent across windows despite deep-block activation blow-up.")

    # Original Dobi-SVD stage hyperparameters (defaults mirror dobisvd.py). Only used with
    # --dobi_svd_sliding.
    parser.add_argument("--dobi_train_steps", type=int, default=100,
                        help="Stage-1 gamma optimization steps")
    parser.add_argument("--dobi_lr", type=float, default=2.0, help="Stage-1 Adam lr for gamma")
    parser.add_argument("--dobi_min_lr", type=float, default=0.5, help="Stage-1 cosine floor lr")
    parser.add_argument("--dobi_beta", type=float, default=10.0, help="sharpness of the tanh rank gate")
    parser.add_argument("--dobi_lambda_reg", type=float, default=50.0, help="weight on the compression-ratio loss")
    parser.add_argument("--dobi_lambda_val", type=float, default=1e-3, help="weight on the gamma-range penalty")
    parser.add_argument("--dobi_ipca_accum", type=int, default=4,
                        help="# forward batches of V accumulated before each IPCA partial_fit")
    args = parser.parse_args()

    seed_everything(GLOBAL_SEED)
    models = [m.strip() for m in args.model_filter.split(",")] if args.model_filter else ["deit_base_patch16_224"]
    ratios = [float(r) for r in args.ratio.split(",")]
    ce_last_windows_list = [int(x) for x in args.ce_last_windows.split(",") if x.strip() != ""]

    # Namespace consumed by dobisvd.train_gamma / dobisvd.ipca_weight_update.
    dobi_args = argparse.Namespace(
        train_steps=args.dobi_train_steps,
        lr=args.dobi_lr,
        min_lr=args.dobi_min_lr,
        beta=args.dobi_beta,
        lambda_reg=args.dobi_lambda_reg,
        lambda_val=args.dobi_lambda_val,
        ipca_accum=args.dobi_ipca_accum,
    )

    if args.run_sliding_window:
        eval_vit_sliding_window_train_rank(
            models, ratios, args.out_dir,
            whitening=args.whitening,
            window_size=args.window_size,
            rank_search_window_size=args.rank_search_window_size,
            stride_size=args.stride_size,
            epochs=args.epochs,
            rank_search_epochs=args.rank_search_epochs,
            calib_size=args.calib_size,
            lambda_rank=args.lambda_rank,
            lambda_lr=args.lambda_lr,
            lr=args.lr,
            rank_lr=args.rank_lr,
            use_dobi=args.dobi_svd_sliding,
            run_global_search=args.run_global_search,
            run_ipca_update=args.run_ipca_update,
            stage2_end_to_end=args.stage2_end_to_end,
            use_cayley=args.cayley,
            freeze_s=args.freeze_s,
            skip_rank_search=args.skip_rank_search,
            dobi_args=dobi_args,
            ce_last_windows_list=ce_last_windows_list,
            kd_temperature=args.kd_temperature,
            ce_rank_search=args.ce_rank_search,
            stage2_decay=args.stage2_decay,
            stage2_temperature=args.stage2_temperature,
            rel_mse=args.rel_mse
        )
