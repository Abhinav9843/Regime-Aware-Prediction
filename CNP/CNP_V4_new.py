import math
import random
from typing import Dict, Tuple, List, Any, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== Device utils =====================
def resolve_device(device: str = "auto") -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def ensure_tensor(x: Any, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


# ===================== Repro =====================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== Tiny MLP factory =====================
def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int,
             activation: str = "ReLU", dropout: float = 0.0) -> nn.Sequential:
    act = getattr(nn, activation)
    layers: List[nn.Module] = []
    for i in range(num_layers):
        in_dim = input_dim if i == 0 else hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

def softplus_eps(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.softplus(x) + eps


# ===================== Student-t Likelihood =====================
def student_t_params_from_decoder_output(raw: torch.Tensor, output_dim: int,
                                         min_nu: float = 2.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, log_scale, nu_pre = raw.chunk(3, dim=-1)
    scale = softplus_eps(log_scale)
    nu = min_nu + softplus_eps(nu_pre)  # ensures nu > min_nu (>2)
    return mean, scale, nu

def student_t_nll(y: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.StudentT(df=nu, loc=mean, scale=scale)
    return (-dist.log_prob(y)).mean()

def student_t_cdf(y: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.StudentT(df=nu, loc=mean, scale=scale)
    return dist.cdf(y)

def student_t_quantiles(mean: torch.Tensor, scale: torch.Tensor, nu: torch.Tensor,
                        alpha: float, max_iter: int = 60) -> torch.Tensor:
    """
    Numeric inverse-CDF via bisection. Stays on GPU if mean/scale/nu are on GPU.
    """
    dist = torch.distributions.StudentT(df=nu, loc=mean, scale=scale)
    alpha_t = torch.as_tensor(alpha, device=mean.device, dtype=mean.dtype)

    lo = mean - 1e3 * scale
    hi = mean + 1e3 * scale
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        c = dist.cdf(mid)
        lo = torch.where(c < alpha_t, mid, lo)
        hi = torch.where(c < alpha_t, hi, mid)
    return 0.5 * (lo + hi)


# ===================== Attention & CNP blocks =====================
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return weights @ v

def encode_context(x_ctx: torch.Tensor, y_ctx: torch.Tensor, encoder_net: nn.Module) -> torch.Tensor:
    return encoder_net(torch.cat([x_ctx, y_ctx], dim=-1))

def cross_attention(x_tgt: torch.Tensor, x_ctx: torch.Tensor, r_ctx: torch.Tensor,
                    key_net: nn.Module, value_net: nn.Module, query_net: nn.Module) -> torch.Tensor:
    k = key_net(x_ctx)
    v = value_net(r_ctx)
    q = query_net(x_tgt)
    return attention(q, k, v)

def decode_student_t(r: torch.Tensor, x_target: torch.Tensor, decoder_net: nn.Module, output_dim: int,
                     min_nu: float = 2.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = decoder_net(torch.cat([r, x_target], dim=-1))
    return student_t_params_from_decoder_output(raw, output_dim=output_dim, min_nu=min_nu)


# ===================== KNN-biased context selection (GPU) =====================
def choose_knn_context_indices(x_batch: torch.Tensor, num_ctx: int, pivots: int = 4) -> torch.Tensor:
    """
    GPU version. Returns indices on x_batch.device.
    NOTE: O(B^2) via cdist. Can be expensive.
    """
    device = x_batch.device
    B = x_batch.shape[0]
    pivots = min(pivots, B)
    pivot_idxs = torch.linspace(0, B - 1, pivots, device=device).round().long()

    with torch.no_grad():
        dists = torch.cdist(x_batch, x_batch, p=2)  # (B,B)
        k_each = max(1, math.ceil(num_ctx / pivots))
        ctx_sets = []
        for p in pivot_idxs:
            nn_idx = torch.topk(-dists[p], k=min(B, k_each), largest=True).indices
            ctx_sets.append(nn_idx)

        ctx_idx = torch.unique(torch.cat(ctx_sets, dim=0))
        if ctx_idx.numel() > num_ctx:
            pivot_d = dists[pivot_idxs][:, ctx_idx]
            scores = pivot_d.min(dim=0).values
            keep = torch.topk(-scores, k=num_ctx, largest=True).indices
            ctx_idx = ctx_idx[keep]
        ctx_idx = torch.sort(ctx_idx).values
    return ctx_idx


# ===================== Per-cluster scaling (GPU / torch) =====================
def fit_cluster_scalers_torch(X: torch.Tensor, Y: torch.Tensor, cluster_ids: torch.Tensor,
                              eps: float = 1e-8) -> Dict[int, Dict[str, torch.Tensor]]:
    scalers: Dict[int, Dict[str, torch.Tensor]] = {}
    for k in torch.unique(cluster_ids).tolist():
        k_int = int(k)
        idx = (cluster_ids == k_int)
        Xk = X[idx]
        Yk = Y[idx]
        mu_x = Xk.mean(dim=0)
        std_x = Xk.std(dim=0, unbiased=False) + eps
        mu_y = Yk.mean(dim=0)
        std_y = Yk.std(dim=0, unbiased=False) + eps
        scalers[k_int] = {"mu_x": mu_x, "std_x": std_x, "mu_y": mu_y, "std_y": std_y}
    return scalers

def apply_scaler_torch(Z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (Z - mu) / std

def invert_scaler_torch(Zs: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return Zs * std + mu


# ===================== Train one CNP (Student-t) per cluster (GPU) =====================
def train_cnp_single_cluster(
    X: torch.Tensor, Y: torch.Tensor,
    hidden_dim: int = 128, num_layers: int = 3,
    activation: str = "ReLU", dropout: float = 0.1,
    lr: float = 1e-3, weight_decay: float = 1e-4,
    epochs: int = 300, batch_size: int = 128,
    min_ctx: int = 16, ctx_frac: float = 0.5, pivots: int = 4,
    grad_clip: float = 1.0, val_frac: float = 0.2,
    patience: int = 20
) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, Dict[str, List[float]]]:
    """
    X, Y must already be on device (GPU if available).
    """
    device = X.device
    N, Dx = X.shape
    Dy = Y.shape[1]

    encoder_net = make_mlp(Dx + Dy, hidden_dim, hidden_dim, num_layers, activation, dropout).to(device)
    key_net     = make_mlp(Dx, hidden_dim, hidden_dim, 2, activation, dropout).to(device)
    value_net   = make_mlp(hidden_dim, hidden_dim, hidden_dim, 2, activation, dropout).to(device)
    query_net   = make_mlp(Dx, hidden_dim, hidden_dim, 2, activation, dropout).to(device)
    decoder_net = make_mlp(hidden_dim + Dx, hidden_dim, Dy * 3, num_layers, activation, dropout).to(device)

    params = list(encoder_net.parameters()) + list(key_net.parameters()) + \
             list(value_net.parameters()) + list(query_net.parameters()) + \
             list(decoder_net.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    # torch-only split
    perm_all = torch.randperm(N, device=device)
    n_val = max(1, int(val_frac * N))
    val_idx = perm_all[:n_val]
    tr_idx  = perm_all[n_val:] if n_val < N else perm_all[:]

    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    best_val = float("inf")
    best_state = None
    history = {"train_nll": [], "val_nll": []}

    for epoch in range(1, epochs + 1):
        encoder_net.train(); key_net.train(); value_net.train(); query_net.train(); decoder_net.train()

        perm = torch.randperm(X_tr.shape[0], device=device)
        total_loss = 0.0
        for i in range(0, X_tr.shape[0], batch_size):
            bidx = perm[i:i + batch_size]
            x_batch = X_tr[bidx]
            y_batch = Y_tr[bidx]
            B = x_batch.shape[0]
            if B < 2:
                continue

            num_ctx = max(min_ctx, int(ctx_frac * B))
            ctx_idx = choose_knn_context_indices(x_batch, num_ctx=num_ctx, pivots=pivots)

            x_ctx, y_ctx = x_batch[ctx_idx], y_batch[ctx_idx]
            r_ctx = encode_context(x_ctx, y_ctx, encoder_net)
            r_tgt = cross_attention(x_batch, x_ctx, r_ctx, key_net, value_net, query_net)
            mean, scale, nu = decode_student_t(r_tgt, x_batch, decoder_net, output_dim=Dy, min_nu=2.1)

            nll = student_t_nll(y_batch, mean, scale, nu)

            optimizer.zero_grad(set_to_none=True)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            total_loss += nll.detach().item() * B

        train_loss = total_loss / max(1, X_tr.shape[0])

        # validation
        encoder_net.eval(); key_net.eval(); value_net.eval(); query_net.eval(); decoder_net.eval()
        with torch.no_grad():
            if X_val.shape[0] > 1:
                num_ctx_val = max(min_ctx, int(ctx_frac * X_val.shape[0]))
                ctx_idx_val = choose_knn_context_indices(X_val, num_ctx=num_ctx_val, pivots=pivots)
                x_ctx_v, y_ctx_v = X_val[ctx_idx_val], Y_val[ctx_idx_val]
                r_ctx_v = encode_context(x_ctx_v, y_ctx_v, encoder_net)
                r_tgt_v = cross_attention(X_val, x_ctx_v, r_ctx_v, key_net, value_net, query_net)
                mean_v, scale_v, nu_v = decode_student_t(r_tgt_v, X_val, decoder_net, output_dim=Dy, min_nu=2.1)
                val_loss = student_t_nll(Y_val, mean_v, scale_v, nu_v).detach().item()
            else:
                val_loss = train_loss

        history["train_nll"].append(train_loss)
        history["val_nll"].append(val_loss)

        # early stopping (keep best state on-device)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder_net.state_dict().items()},
                "key":     {k: v.detach().clone() for k, v in key_net.state_dict().items()},
                "value":   {k: v.detach().clone() for k, v in value_net.state_dict().items()},
                "query":   {k: v.detach().clone() for k, v in query_net.state_dict().items()},
                "decoder": {k: v.detach().clone() for k, v in decoder_net.state_dict().items()},
            }

        best_epoch = int(torch.tensor(history["val_nll"], device=device).argmin().item())
        if (epoch - (best_epoch + 1)) >= patience:
            break

    if best_state is not None:
        encoder_net.load_state_dict(best_state["encoder"])
        key_net.load_state_dict(best_state["key"])
        value_net.load_state_dict(best_state["value"])
        query_net.load_state_dict(best_state["query"])
        decoder_net.load_state_dict(best_state["decoder"])

    return encoder_net, key_net, value_net, query_net, decoder_net, history


# ===================== Predict for Student-t CNP (GPU) =====================
@torch.no_grad()
def predict_cnp_student_t(
    x_ctx: Any, y_ctx: Any, x_tgt: Any,
    encoder_net: nn.Module, key_net: nn.Module, value_net: nn.Module,
    query_net: nn.Module, decoder_net: nn.Module,
    device: str = "auto",
    return_numpy: bool = True
) -> Tuple[Any, Any, Any]:
    """
    Works with numpy or torch inputs; runs on GPU if available and models are on that device.
    Returns (mean, scale, nu) as torch or numpy.
    """
    dev = resolve_device(device)
    # ensure models are on dev
    encoder_net = encoder_net.to(dev)
    key_net = key_net.to(dev)
    value_net = value_net.to(dev)
    query_net = query_net.to(dev)
    decoder_net = decoder_net.to(dev)

    x_ctx_t = ensure_tensor(x_ctx, dev)
    y_ctx_t = ensure_tensor(y_ctx, dev)
    x_tgt_t = ensure_tensor(x_tgt, dev)

    r_ctx = encode_context(x_ctx_t, y_ctx_t, encoder_net)
    r_tgt = cross_attention(x_tgt_t, x_ctx_t, r_ctx, key_net, value_net, query_net)
    mean, scale, nu = decode_student_t(r_tgt, x_tgt_t, decoder_net, output_dim=y_ctx_t.shape[1], min_nu=2.1)

    if return_numpy:
        return mean.detach().cpu().numpy(), scale.detach().cpu().numpy(), nu.detach().cpu().numpy()
    return mean, scale, nu


# ===================== Calibration helpers (GPU) =====================
@torch.no_grad()
def interval_coverage_student_t(
    y_true: Any, mean: Any, scale: Any, nu: Any,
    alphas: List[float] = (0.05, 0.2),
    device: str = "auto"
) -> Dict[str, float]:
    """
    Computes empirical coverage for central (1-alpha) intervals using Student-t inverse CDF.
    Runs on GPU if available.
    """
    dev = resolve_device(device)
    y_t = ensure_tensor(y_true, dev)
    mean_t = ensure_tensor(mean, dev)
    scale_t = ensure_tensor(scale, dev)
    nu_t = ensure_tensor(nu, dev)

    coverages: Dict[str, float] = {}
    for a in alphas:
        lo = student_t_quantiles(mean_t, scale_t, nu_t, a / 2.0)
        hi = student_t_quantiles(mean_t, scale_t, nu_t, 1.0 - a / 2.0)
        inside = ((y_t >= lo) & (y_t <= hi)).float().mean().item()
        coverages[f"cov@{int((1.0 - a) * 100)}"] = float(inside)
    return coverages

@torch.no_grad()
def pit_values_student_t(
    y_true: Any, mean: Any, scale: Any, nu: Any,
    device: str = "auto",
    return_numpy: bool = True
) -> Any:
    """
    PIT = F_Y(y_true). Runs on GPU if available (depends on StudentT.cdf support).
    """
    dev = resolve_device(device)
    y_t = ensure_tensor(y_true, dev)
    mean_t = ensure_tensor(mean, dev)
    scale_t = ensure_tensor(scale, dev)
    nu_t = ensure_tensor(nu, dev)

    pit = student_t_cdf(y_t, mean_t, scale_t, nu_t)
    if return_numpy:
        return pit.detach().cpu().numpy()
    return pit


# ===================== Training wrapper (per cluster) - GPU / torch =====================
def train_cnp_per_cluster(
    X: Any, Y: Any, cluster_ids: Any,
    hidden_dim: int = 128, num_layers: int = 3, activation: str = "ReLU",
    dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4,
    epochs: int = 300, batch_size: int = 128, min_ctx: int = 16,
    ctx_frac: float = 0.5, pivots: int = 4, grad_clip: float = 1.0,
    val_frac: float = 0.2, patience: int = 20,
    device: str = "auto"
) -> Tuple[Dict[int, Tuple], Dict[int, Dict[str, torch.Tensor]]]:
    """
    Full-GPU (torch-only) training wrapper.
    Returns:
      models[k] = (encoder,key,value,query,decoder,history)
      scalers[k] = {mu_x,std_x,mu_y,std_y} (GPU tensors)
    """
    dev = resolve_device(device)
    X_t = ensure_tensor(X, dev)
    Y_t = ensure_tensor(Y, dev)
    C_t = ensure_tensor(cluster_ids, dev, dtype=torch.float32).long()

    scalers = fit_cluster_scalers_torch(X_t, Y_t, C_t)
    models: Dict[int, Tuple] = {}

    for k in sorted([int(v) for v in torch.unique(C_t).tolist()]):
        idx = (C_t == k)
        Xk_raw = X_t[idx]
        Yk_raw = Y_t[idx]

        Xk = apply_scaler_torch(Xk_raw, scalers[k]["mu_x"], scalers[k]["std_x"])
        Yk = apply_scaler_torch(Yk_raw, scalers[k]["mu_y"], scalers[k]["std_y"])

        enc, key, val, qry, dec, hist = train_cnp_single_cluster(
            Xk, Yk,
            hidden_dim=hidden_dim, num_layers=num_layers,
            activation=activation, dropout=dropout,
            lr=lr, weight_decay=weight_decay,
            epochs=epochs, batch_size=batch_size,
            min_ctx=min_ctx, ctx_frac=ctx_frac, pivots=pivots,
            grad_clip=grad_clip, val_frac=val_frac,
            patience=patience
        )
        models[k] = (enc, key, val, qry, dec, hist)

    return models, scalers


# ===================== HPO: Multi-task Successive Halving (GPU) =====================
def _prepare_cluster_tensors(X: Any, Y: Any, cluster_ids: Any, device: torch.device):
    X_t = ensure_tensor(X, device)
    Y_t = ensure_tensor(Y, device)
    C_t = ensure_tensor(cluster_ids, device, dtype=torch.float32).long()

    keys = sorted([int(v) for v in torch.unique(C_t).tolist()])
    per_cluster: Dict[int, Dict[str, torch.Tensor]] = {}
    for k in keys:
        idx = (C_t == k)
        per_cluster[k] = {
            "X_raw": X_t[idx],
            "Y_raw": Y_t[idx],
            "n": torch.tensor(int(idx.sum().item()), device=device, dtype=torch.float32)
        }
    return per_cluster, keys

def _scale_all_clusters_torch(per_cluster: Dict[int, Dict[str, torch.Tensor]], eps: float = 1e-8):
    """
    Computes and caches X_scaled/Y_scaled per cluster on GPU; returns normalized weights (torch).
    """
    device = next(iter(per_cluster.values()))["X_raw"].device
    weights = {}
    total = 0.0
    for k, obj in per_cluster.items():
        Xk = obj["X_raw"]
        Yk = obj["Y_raw"]
        mu_x = Xk.mean(dim=0)
        std_x = Xk.std(dim=0, unbiased=False) + eps
        mu_y = Yk.mean(dim=0)
        std_y = Yk.std(dim=0, unbiased=False) + eps

        obj["mu_x"] = mu_x; obj["std_x"] = std_x
        obj["mu_y"] = mu_y; obj["std_y"] = std_y
        obj["X_scaled"] = (Xk - mu_x) / std_x
        obj["Y_scaled"] = (Yk - mu_y) / std_y

        weights[k] = obj["n"]
        total += float(obj["n"].item())

    w_vec = torch.tensor([float(weights[k].item()) for k in sorted(weights.keys())],
                         device=device, dtype=torch.float32)
    w_vec = w_vec / max(w_vec.sum().item(), 1.0)
    return w_vec  # aligned with sorted(per_cluster.keys())

def _score_config_on_subset(
    cfg: Dict[str, Any],
    per_cluster: Dict[int, Dict[str, torch.Tensor]],
    subset_keys: List[int],
    weights_vec: torch.Tensor,   # aligned with sorted(all_keys)
    all_keys_sorted: List[int],
    epochs: int, batch_size: int, min_ctx: int, ctx_frac: float,
    device: torch.device
) -> float:
    """
    Weighted mean of best val NLL across subset clusters. All training on GPU.
    """
    key_to_weight_idx = {k: i for i, k in enumerate(all_keys_sorted)}
    losses = []
    wts = []
    for k in subset_keys:
        Xk = per_cluster[k]["X_scaled"]
        Yk = per_cluster[k]["Y_scaled"]

        _, _, _, _, _, hist = train_cnp_single_cluster(
            Xk, Yk,
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            activation=str(cfg["activation"]),
            dropout=float(cfg["dropout"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            epochs=int(epochs),
            batch_size=int(batch_size),
            min_ctx=int(min_ctx),
            ctx_frac=float(ctx_frac),
            pivots=int(cfg["pivots"]),
            grad_clip=1.0,
            val_frac=0.2,
            patience=max(5, min(int(epochs // 5), 20)),
        )
        best = min(hist["val_nll"]) if len(hist["val_nll"]) else float("inf")
        losses.append(best)
        wts.append(float(weights_vec[key_to_weight_idx[k]].item()))

    # normalize weights within subset
    w_sum = sum(wts) if wts else 1.0
    wts = [w / w_sum for w in wts]
    return float(sum(w * l for w, l in zip(wts, losses)))

def multitask_successive_halving_search(
    X: Any, Y: Any, cluster_ids: Any,
    device: str = "auto",
    hidden_dim_choices=(64, 96, 128),
    num_layers_choices=(2, 3, 4),
    dropout_choices=(0.0, 0.1, 0.2),
    activation_choices=("ReLU", "SELU", "Tanh", "Softplus", "LeakyReLU", "PReLU", "Sigmoid"),
    lr_choices=(5e-4, 1e-3, 2e-3),
    weight_decay_choices=(0.0, 1e-5, 1e-4),
    pivots_choices=(2, 4),
    eta: int = 3,
    rung_setups=((20, 64, 0.4), (80, 96, 0.7), (200, 128, 1.0)),  # (epochs, batch, frac_clusters)
    n_initial_configs: int = 27,
    min_ctx: int = 8,
    ctx_frac: float = 0.5,
    seed: int = 123
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Full-GPU successive halving.
    Returns: best_cfg, logs
    """
    dev = resolve_device(device)
    rng = random.Random(seed)

    per_cluster, all_keys = _prepare_cluster_tensors(X, Y, cluster_ids, dev)
    all_keys_sorted = sorted(all_keys)
    weights_vec = _scale_all_clusters_torch(per_cluster)  # aligned with sorted(all_keys)

    def sample_config():
        return {
            "hidden_dim": rng.choice(hidden_dim_choices),
            "num_layers": rng.choice(num_layers_choices),
            "dropout": rng.choice(dropout_choices),
            "activation": rng.choice(activation_choices),
            "lr": rng.choice(lr_choices),
            "weight_decay": rng.choice(weight_decay_choices),
            "pivots": rng.choice(pivots_choices),
        }

    configs = [sample_config() for _ in range(n_initial_configs)]
    survivors = list(range(len(configs)))
    logs: List[Dict[str, Any]] = []

    for rung_idx, (epochs, batch_size, frac_clusters) in enumerate(rung_setups):
        n_use = max(1, int(math.ceil(frac_clusters * len(all_keys_sorted))))
        subset_keys = all_keys_sorted[:n_use]  # deterministic; can randomize if you want
        rung_scores = []

        for ci in survivors:
            cfg = configs[ci]
            score = _score_config_on_subset(
                cfg, per_cluster, subset_keys,
                weights_vec=weights_vec,
                all_keys_sorted=all_keys_sorted,
                epochs=int(epochs),
                batch_size=int(batch_size),
                min_ctx=int(min_ctx),
                ctx_frac=float(ctx_frac),
                device=dev,
            )
            rung_scores.append((ci, score))
            logs.append({
                "rung": rung_idx,
                "config_idx": ci,
                "score": score,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "clusters_used": int(n_use),
                "config": deepcopy(cfg),
            })

        rung_scores.sort(key=lambda t: t[1])  # lower is better

        if rung_idx < len(rung_setups) - 1:
            keep = max(1, len(rung_scores) // eta)
            survivors = [ci for (ci, _) in rung_scores[:keep]]
        else:
            best_ci = rung_scores[0][0]
            return deepcopy(configs[best_ci]), logs

    return deepcopy(configs[survivors[0]]), logs


# ===================== Gating over clusters (GPU / torch) =====================
def fit_gaussian_gater_per_cluster_torch(X: torch.Tensor, cluster_ids: torch.Tensor, eps: float = 1e-6):
    params: Dict[int, Dict[str, torch.Tensor]] = {}
    priors: Dict[int, float] = {}
    N = X.shape[0]
    for k in sorted([int(v) for v in torch.unique(cluster_ids).tolist()]):
        idx = (cluster_ids == k)
        Xk = X[idx]
        mu = Xk.mean(dim=0)
        var = Xk.var(dim=0, unbiased=False) + eps
        params[k] = {"mu": mu, "var": var}
        priors[k] = float(Xk.shape[0]) / float(N)
    return params, priors

def gater_posterior_probs_torch(x_star: torch.Tensor, g_params, g_priors) -> Tuple[List[int], torch.Tensor]:
    """
    Returns (keys, weights) where weights is a GPU tensor aligned with keys.
    """
    Dx = x_star.shape[0]
    keys = sorted(g_params.keys())

    logw = []
    for k in keys:
        mu = g_params[k]["mu"]
        var = g_params[k]["var"]
        # diagonal Gaussian logpdf + log prior
        log_det = 0.5 * torch.sum(torch.log(var))
        quad = 0.5 * torch.sum((x_star - mu) ** 2 / var)
        logp = -0.5 * Dx * math.log(2.0 * math.pi) - log_det - quad + math.log(g_priors[k] + 1e-12)
        logw.append(logp)
    logw = torch.stack(logw, dim=0)
    w = torch.softmax(logw, dim=0)
    return keys, w


# ===================== Context selection in cluster (GPU / torch) =====================
@torch.no_grad()
def select_context_knn_in_cluster_torch(X_cluster: torch.Tensor, Y_cluster: torch.Tensor,
                                        x_star: torch.Tensor, k_ctx: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = X_cluster.mean(dim=0)
    std = X_cluster.std(dim=0, unbiased=False) + 1e-8
    Xz = (X_cluster - mu) / std
    xz = (x_star - mu) / std
    d = torch.norm(Xz - xz[None, :], dim=1)
    k = min(k_ctx, X_cluster.shape[0])
    idx = torch.topk(-d, k=k, largest=True).indices
    return X_cluster[idx], Y_cluster[idx]


# ===================== Mixture utilities (GPU / torch) =====================

def t_var_from_scale_nu_torch(scale: torch.Tensor, nu: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    nu_safe = torch.clamp(nu, min=2.0 + eps)
    return (nu_safe / (nu_safe - 2.0)) * (scale ** 2)


def mixture_mean_var_student_t_torch(mus: List[torch.Tensor], scales: List[torch.Tensor], nus: List[torch.Tensor],
                                     weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mus/scales/nus: list of (Dy,) tensors
    weights: (K,) tensor
    Returns (mu_mix, var_mix) tensors (Dy,)
    """
    K = len(mus)
    mu_stack = torch.stack([m.reshape(-1) for m in mus], dim=0)      # (K, Dy)
    sc_stack = torch.stack([s.reshape(-1) for s in scales], dim=0)   # (K, Dy)
    nu_stack = torch.stack([n.reshape(-1) for n in nus], dim=0)      # (K, Dy)

    w = weights.reshape(K, 1)  # (K,1)
    mu_mix = torch.sum(w * mu_stack, dim=0)

    var_k = t_var_from_scale_nu_torch(sc_stack, nu_stack)            # (K,Dy)
    second_mom = torch.sum(w * (var_k + mu_stack ** 2), dim=0)
    var_mix = second_mom - mu_mix ** 2
    var_mix = torch.clamp(var_mix, min=0.0)
    return mu_mix, var_mix

@torch.no_grad()
def mixture_quantile_student_t_torch(alpha: float, mus: List[torch.Tensor], scales: List[torch.Tensor],
                                     nus: List[torch.Tensor], weights: torch.Tensor,
                                     max_iter: int = 80) -> torch.Tensor:
    """
    Numeric inverse-CDF for Student-t mixture, all on GPU (assuming StudentT.cdf runs on GPU in your build).
    Returns (Dy,) tensor.
    """
    K = len(mus)
    mu_t = torch.stack([m.reshape(-1) for m in mus], dim=0)      # (K, Dy)
    sc_t = torch.stack([s.reshape(-1) for s in scales], dim=0)   # (K, Dy)
    nu_t = torch.stack([n.reshape(-1) for n in nus], dim=0)      # (K, Dy)
    w_t = weights.reshape(K, 1)                                  # (K,1)

    lo = (mu_t.min(dim=0).values - 1000.0 * sc_t.max(dim=0).values).clone()
    hi = (mu_t.max(dim=0).values + 1000.0 * sc_t.max(dim=0).values).clone()
    target = torch.tensor(alpha, device=mu_t.device, dtype=mu_t.dtype)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)  # (Dy,)
        Fm = 0.0
        for k in range(K):
            dist = torch.distributions.StudentT(df=nu_t[k], loc=mu_t[k], scale=sc_t[k])
            Fm = Fm + w_t[k] * dist.cdf(mid)
        lo = torch.where(Fm < target, mid, lo)
        hi = torch.where(Fm < target, hi, mid)

    return 0.5 * (lo + hi)


# ===================== Single-target mixture prediction (GPU / torch) =====================
@torch.no_grad()
def predict_single_target_mixture(
    x_star_raw, X, Y, cluster_ids,
    models: Dict[int, Tuple], scalers: Dict[int, Dict[str, torch.Tensor]],
    k_ctx: int = 64,
    device: str = "auto",
    return_numpy: bool = True
) -> dict:
    """
    Fully torch/GPU pipeline. Optionally converts outputs to numpy at the end.
    """
    dev = resolve_device(device)
    X_t = ensure_tensor(X, dev)
    Y_t = ensure_tensor(Y, dev)
    C_t = ensure_tensor(cluster_ids, dev).long()
    x_star_t = ensure_tensor(x_star_raw, dev).reshape(-1)

    # gater on raw X
    g_params, g_priors = fit_gaussian_gater_per_cluster_torch(X_t, C_t)
    keys, weights = gater_posterior_probs_torch(x_star_t, g_params, g_priors)  # weights: (K,) torch

    mus, scales, nus = [], [], []
    per_cluster = {}

    for j, k in enumerate(keys):
        idx = (C_t == k)
        Xk_raw = X_t[idx]
        Yk_raw = Y_t[idx]

        # context near x*
        Xc_raw, Yc_raw = select_context_knn_in_cluster_torch(Xk_raw, Yk_raw, x_star_t, k_ctx=k_ctx)

        # scale for model I/O
        Xc = apply_scaler_torch(Xc_raw, scalers[k]["mu_x"], scalers[k]["std_x"])
        Yc = apply_scaler_torch(Yc_raw, scalers[k]["mu_y"], scalers[k]["std_y"])
        x_star = apply_scaler_torch(x_star_t, scalers[k]["mu_x"], scalers[k]["std_x"]).reshape(1, -1)

        enc, key, val, qry, dec, _ = models[k]

        # Ensure model + inputs on same device
        model_dev = next(enc.parameters()).device
        if model_dev != dev:
            # You trained on a different device. This keeps things consistent.
            dev = model_dev
            Xc = Xc.to(dev); Yc = Yc.to(dev); x_star = x_star.to(dev)
            weights = weights.to(dev)
            x_star_t = x_star_t.to(dev)

        mean_s, scale_s, nu_s = predict_cnp_student_t(Xc, Yc, x_star, enc, key, val, qry, dec, return_numpy = False)

        # back to original Y units
        mean = invert_scaler_torch(mean_s, scalers[k]["mu_y"], scalers[k]["std_y"]).reshape(-1)
        scale = (scale_s * scalers[k]["std_y"]).reshape(-1)
        nu = nu_s.reshape(-1)

        mus.append(mean)
        scales.append(scale)
        nus.append(nu)

        per_cluster[k] = {
            "mean": mean,
            "scale": scale,
            "nu": nu,
            "weight": weights[j]
        }

    mu_mix, var_mix = mixture_mean_var_student_t_torch(mus, scales, nus, weights)

    # build result
    if return_numpy:
        out = {
            "weights": {k: float(weights[i].detach().cpu()) for i, k in enumerate(keys)},
            "per_cluster": {
                int(k): {
                    "mean": per_cluster[k]["mean"].detach().cpu().numpy(),
                    "scale": per_cluster[k]["scale"].detach().cpu().numpy(),
                    "nu": per_cluster[k]["nu"].detach().cpu().numpy(),
                    "weight": float(per_cluster[k]["weight"].detach().cpu())
                }
                for k in keys
            },
            "mean_mix": mu_mix.detach().cpu().numpy(),
            "var_mix": var_mix.detach().cpu().numpy(),
        }
    else:
        out = {
            "weights": {k: weights[i] for i, k in enumerate(keys)},
            "per_cluster": {int(k): per_cluster[k] for k in keys},
            "mean_mix": mu_mix,
            "var_mix": var_mix,
        }

    return out

def make_synth_regime_data(n_per_cluster: List[int], Dx: int = 4, Dy: int = 1,
                           seed: int = 0, device: str = "auto") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = resolve_device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)

    X_list, Y_list, C_list = [], [], []
    for k, n in enumerate(n_per_cluster):
        W = torch.randn(Dx, Dy, generator=g, device=dev) * 0.8 + (k + 1) * 0.2
        b = torch.randn(Dy, generator=g, device=dev) * 0.5

        Xk = torch.randn(n, Dx, generator=g, device=dev) * (1.0 + 0.2 * k) + (k * 0.5)
        nonlin = torch.tanh(Xk @ W + b) + 0.5 * torch.sin(Xk[:, :1])

        df = 3.0 + 2.0 * k
        scale = 0.5 + 0.2 * k
        noise = torch.distributions.StudentT(df=torch.tensor(df, device=dev), loc=0.0, scale=1.0).sample((n, Dy))
        noise = noise * scale

        Yk = nonlin + noise
        X_list.append(Xk)
        Y_list.append(Yk)
        C_list.append(torch.full((n,), k, dtype=torch.long, device=dev))

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    C = torch.cat(C_list, dim=0)
    return X, Y, C

