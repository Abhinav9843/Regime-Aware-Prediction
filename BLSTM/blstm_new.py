import math
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import os
import torch.nn.functional as F



def find_project_root(start: Path) -> Path:
    """
    Walk upward until we find the project root (marker folders).
    Adjust markers to match your repo structure.
    """
    markers = ["Prediction Result", "Trading_Result", "Regime_Results"]
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if all((p / m).exists() for m in markers):
            return p
    raise FileNotFoundError(
        f"Could not auto-detect project root from {start}. "
        f"Run with --base-dir or set RCNP_BASE_DIR."
    )

def get_base_dir() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.base_dir:
        return Path(args.base_dir).expanduser().resolve()

    env = os.getenv("RCNP_BASE_DIR")
    if env:
        return Path(env).expanduser().resolve()

    # Default: locate relative to this script file
    here = Path(__file__).resolve().parent
    return find_project_root(here)



BASE_DIR = get_base_dir()


# ----------------------------
# Repro / device
# ----------------------------
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Data utilities (2D -> windows)
# ----------------------------
def make_windows_2d(X2d: np.ndarray, y2d: np.ndarray, seq_len: int):
    """
    X2d: [T, F]
    y2d: [T, K]
    Creates:
      Xw[i] = X2d[i : i+seq_len]         -> [seq_len, F]
      yw[i] = y2d[i+seq_len]              -> [K]   (predict next step)
    Returns:
      Xw: [N, seq_len, F], yw: [N, K]
    """
    X2d = np.asarray(X2d, dtype=np.float32)
    y2d = np.asarray(y2d, dtype=np.float32)

    assert X2d.ndim == 2, f"X must be 2D [T,F], got {X2d.shape}"
    assert y2d.ndim == 2, f"y must be 2D [T,K], got {y2d.shape}"
    assert X2d.shape[0] == y2d.shape[0], "X and y must have same T"
    assert seq_len >= 1, "seq_len must be >= 1"
    T, F_ = X2d.shape
    _, K_ = y2d.shape

    N = T - seq_len
    if N <= 0:
        raise ValueError(f"seq_len={seq_len} too large for T={T}. Need T - seq_len > 0.")

    Xw = np.zeros((N, seq_len, F_), dtype=np.float32)
    yw = np.zeros((N, K_), dtype=np.float32)

    for i in range(N):
        Xw[i] = X2d[i : i + seq_len]
        yw[i] = y2d[i + seq_len]

    return Xw, yw


def time_split(Xw, yw, train_frac=0.7, val_frac=0.15):
    """
    Time-ordered split (NO shuffling):
      train: [0 : n_train)
      val:   [n_train : n_train+n_val)
      test:  remainder
    """
    N = Xw.shape[0]
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    X_train, y_train = Xw[:n_train], yw[:n_train]
    X_val, y_val = Xw[n_train : n_train + n_val], yw[n_train : n_train + n_val]
    X_test, y_test = Xw[n_train + n_val :], yw[n_train + n_val :]
    return X_train, y_train, X_val, y_val, X_test, y_test


# ----------------------------
# Standardization
# ----------------------------
def fit_standardizer_X(X_train: np.ndarray):
    """
    X_train: [N, seq_len, F]
    Standardize per feature across N*seq_len
    """
    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sigma = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def transform_X(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return ((X - mu[None, None, :]) / sigma[None, None, :]).astype(np.float32)


def fit_standardizer_y(y_train: np.ndarray):
    """
    y_train: [N, K]
    Standardize per output dimension
    """
    mu = y_train.mean(axis=0)
    sigma = y_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def transform_y(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return ((y - mu[None, :]) / sigma[None, :]).astype(np.float32)


def inverse_y(mu_hat: np.ndarray, var_hat: np.ndarray, y_mu: np.ndarray, y_sigma: np.ndarray):
    """
    If y was standardized:
      y = y_mu + y_sigma * y_std
      Var(y) = (y_sigma^2) * Var(y_std)
    """
    mu = mu_hat * y_sigma[None, :] + y_mu[None, :]
    var = var_hat * (y_sigma[None, :] ** 2)
    return mu, var


# ----------------------------
# Model (NO custom class)
# ----------------------------
def build_bayesian_lstm_modules(
    n_features: int,
    hidden1: int,
    hidden2: int,
    num_layers: int,
    output_dim: int,
    device,
    learn_aleatoric: bool = True,
):
    """
    modules dict:
      lstm1, lstm2
      fc_mu
      fc_logvar (optional)   # predicts log variance per output
    """
    lstm1 = nn.LSTM(
        input_size=n_features,
        hidden_size=hidden1,
        num_layers=num_layers,
        batch_first=True,
    ).to(device)

    lstm2 = nn.LSTM(
        input_size=hidden1,
        hidden_size=hidden2,
        num_layers=num_layers,
        batch_first=True,
    ).to(device)

    fc_mu = nn.Linear(hidden2, output_dim).to(device)

    mods = {"lstm1": lstm1, "lstm2": lstm2, "fc_mu": fc_mu}

    if learn_aleatoric:
        fc_logvar = nn.Linear(hidden2, output_dim).to(device)
        mods["fc_logvar"] = fc_logvar

    return mods


def model_parameters(modules: dict):
    params = []
    for m in modules.values():
        params += list(m.parameters())
    return params


def set_train_mode(modules: dict, train: bool):
    for m in modules.values():
        m.train(train)


def forward_bayesian_lstm(modules: dict, x: torch.Tensor, dropout_p: float, mc_dropout: bool):
    """
    x: [B, seq_len, F]
    Returns:
      mu: [B, K]
      logvar: [B, K] or None
    """
    out1, _ = modules["lstm1"](x)
    out1 = F.dropout(out1, p=dropout_p, training=(mc_dropout or modules["lstm1"].training))

    out2, _ = modules["lstm2"](out1)
    out2 = F.dropout(out2, p=dropout_p, training=(mc_dropout or modules["lstm2"].training))

    last = out2[:, -1, :]  # [B, hidden2]

    mu = modules["fc_mu"](last)  # [B, K]

    logvar = None
    if "fc_logvar" in modules:
        logvar = modules["fc_logvar"](last)  # [B, K]

    return mu, logvar


# ----------------------------
# Losses / metrics
# ----------------------------
def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, clamp=(-10.0, 8.0)):
    """
    Per-dimension heteroscedastic Gaussian NLL:
      0.5 * (logvar + (y-mu)^2 / exp(logvar))
    """
    assert y.shape == mu.shape == logvar.shape, (y.shape, mu.shape, logvar.shape)
    logvar = torch.clamp(logvar, clamp[0], clamp[1])
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (logvar + (y - mu) ** 2 * inv_var)
    return nll.mean()


@torch.no_grad()
def eval_metrics(modules: dict, dropout_p: float, X: np.ndarray, y: np.ndarray, device, learn_aleatoric: bool):
    set_train_mode(modules, False)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    mu, logvar = forward_bayesian_lstm(modules, X_t, dropout_p=dropout_p, mc_dropout=False)
    assert mu.shape == y_t.shape, f"Shape bug: mu {mu.shape} vs y {y_t.shape}"

    mse = F.mse_loss(mu, y_t).item()
    if learn_aleatoric:
        nll = gaussian_nll(y_t, mu, logvar).item()
    else:
        nll = float("nan")
    return mse, nll


# ----------------------------
# Training
# ----------------------------
def iterate_minibatches(X, y, batch_size: int, shuffle: bool = True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        b = idx[start : start + batch_size]
        yield X[b], y[b]


def fit_with_early_stopping(
    modules: dict,
    dropout_p: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int = 10,
    min_epochs: int = 20,
    grad_clip: float = 1.0,
    learn_aleatoric: bool = True,
):
    opt = torch.optim.Adam(model_parameters(modules), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, max_epochs + 1):
        set_train_mode(modules, True)

        for Xb, yb in iterate_minibatches(X_train, y_train, batch_size=batch_size, shuffle=True):
            X_t = torch.tensor(Xb, dtype=torch.float32, device=device)
            y_t = torch.tensor(yb, dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            mu, logvar = forward_bayesian_lstm(modules, X_t, dropout_p=dropout_p, mc_dropout=False)

            # hard shape check (prevents silent broadcasting)
            assert mu.shape == y_t.shape, f"Shape bug: mu {mu.shape} vs y {y_t.shape}"

            if learn_aleatoric:
                loss = gaussian_nll(y_t, mu, logvar)
            else:
                loss = F.mse_loss(mu, y_t)

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model_parameters(modules), grad_clip)
            opt.step()

        # validation
        val_mse, val_nll = eval_metrics(modules, dropout_p, X_val, y_val, device, learn_aleatoric)
        val_score = val_nll if learn_aleatoric else val_mse  # monitor NLL if available

        if epoch <= min_epochs:
            if val_score < best_val:
                best_val = val_score
                best_state = {k: v.state_dict() for k, v in modules.items()}
            continue

        if val_score < best_val - 1e-6:
            best_val = val_score
            best_state = {k: v.state_dict() for k, v in modules.items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        for k in modules.keys():
            modules[k].load_state_dict(best_state[k])

    return best_val


# ----------------------------
# MC Dropout prediction (batched)
# ----------------------------
@torch.no_grad()
def mc_dropout_predict(
    modules: dict,
    dropout_p: float,
    X: np.ndarray,
    device,
    mc_samples: int = 100,
    batch_size: int = 256,
    learn_aleatoric: bool = True,
):
    """
    Returns:
      mean_mu: [N, K]
      total_var: [N, K]   (epistemic + aleatoric if enabled)
    """
    set_train_mode(modules, False)
    N = X.shape[0]

    
    sum_mu = None
    sum_mu2 = None
    sum_alea = None

    for s in range(mc_samples):
        mu_all = []
        alea_all = []

        for start in range(0, N, batch_size):
            xb = X[start : start + batch_size]
            X_t = torch.tensor(xb, dtype=torch.float32, device=device)

            mu, logvar = forward_bayesian_lstm(modules, X_t, dropout_p=dropout_p, mc_dropout=True)
            mu_np = mu.detach().cpu().numpy()
            mu_all.append(mu_np)

            if learn_aleatoric:
                logvar = torch.clamp(logvar, -10.0, 8.0)
                alea = torch.exp(logvar).detach().cpu().numpy()  # [B, K]
                alea_all.append(alea)

        mu_s = np.concatenate(mu_all, axis=0)  # [N, K]
        if sum_mu is None:
            sum_mu = mu_s.copy()
            sum_mu2 = (mu_s ** 2).copy()
        else:
            sum_mu += mu_s
            sum_mu2 += (mu_s ** 2)

        if learn_aleatoric:
            alea_s = np.concatenate(alea_all, axis=0)  # [N, K]
            if sum_alea is None:
                sum_alea = alea_s.copy()
            else:
                sum_alea += alea_s

    mean_mu = sum_mu / mc_samples
    epistemic_var = (sum_mu2 / mc_samples) - (mean_mu ** 2)
    epistemic_var = np.maximum(epistemic_var, 0.0)

    if learn_aleatoric:
        aleatoric_var = sum_alea / mc_samples
        total_var = epistemic_var + aleatoric_var
    else:
        total_var = epistemic_var

    return mean_mu.astype(np.float32), total_var.astype(np.float32)

"""
# ----------------------------
# Example loader (replace this)
# ----------------------------
def load_your_data():
    """
    #Replace this with your real loading.
    #Must return:
    #  X2d: [T, 248]
    #  y2d: [T, 48]
    """
    # Dummy synthetic example (same shapes idea)
    T = 1453
    F_ = 248
    K_ = 48
    rng = np.random.default_rng(123)

    X = rng.standard_normal((T, F_)).astype(np.float32)

    # create y with some structure + noise
    base = 0.7 * np.tanh(X[:, :1])  # [T,1]
    seasonal = np.sin(np.linspace(0, 10 * np.pi, T)).astype(np.float32)[:, None]
    curve = np.linspace(-1, 1, K_).astype(np.float32)[None, :]  # [1,K]
    y = (base @ np.ones((1, K_), np.float32)) + 0.3 * seasonal @ np.ones((1, K_), np.float32) + 0.2 * curve
    y += 0.15 * rng.standard_normal((T, K_)).astype(np.float32)
    return X, y


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(123)
    device = get_device()
    print("Device:", device)

    # 1) Load your 2D data
    X2d, y2d = load_your_data()
    print("Raw X:", X2d.shape, "Raw y:", y2d.shape)

    # 2) Choose seq_len
    # If you truly want "no history", set seq_len=1 (still a seq_len, but minimal).
    seq_len = 1   # try 7, 14, 30, 60; OR set 1 if you insist
    Xw, yw = make_windows_2d(X2d, y2d, seq_len=seq_len)
    print("Windowed X:", Xw.shape, "Windowed y:", yw.shape)

    # 3) Time split
    X_train, y_train, X_val, y_val, X_test, y_test = time_split(Xw, yw, train_frac=0.7, val_frac=0.15)
    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    # 4) Standardize X and y (recommended)
    X_mu, X_sigma = fit_standardizer_X(X_train)
    X_train_s = transform_X(X_train, X_mu, X_sigma)
    X_val_s = transform_X(X_val, X_mu, X_sigma)
    X_test_s = transform_X(X_test, X_mu, X_sigma)

    y_mu, y_sigma = fit_standardizer_y(y_train)
    y_train_s = transform_y(y_train, y_mu, y_sigma)
    y_val_s = transform_y(y_val, y_mu, y_sigma)
    y_test_s = transform_y(y_test, y_mu, y_sigma)

    # 5) Build model
    learn_aleatoric = True  # set False if you only want MC-dropout epistemic uncertainty
    modules = build_bayesian_lstm_modules(
        n_features=X_train_s.shape[-1],
        hidden1=128,
        hidden2=64,
        num_layers=1,
        output_dim=y_train_s.shape[-1],
        device=device,
        learn_aleatoric=learn_aleatoric,
    )

    # 6) Train
    dropout_p = 0.2
    best = fit_with_early_stopping(
        modules=modules,
        dropout_p=dropout_p,
        X_train=X_train_s,
        y_train=y_train_s,
        X_val=X_val_s,
        y_val=y_val_s,
        device=device,
        lr=1e-3,
        weight_decay=1e-6,
        batch_size=64,
        max_epochs=200,
        patience=12,
        min_epochs=25,
        grad_clip=1.0,
        learn_aleatoric=learn_aleatoric,
    )
    print("Best monitored val score:", best)

    # 7) Evaluate (standardized)
    test_mse_s, test_nll_s = eval_metrics(modules, dropout_p, X_test_s, y_test_s, device, learn_aleatoric)
    print(f"Test MSE (std-space): {test_mse_s:.6f}")
    if learn_aleatoric:
        print(f"Test NLL (std-space): {test_nll_s:.6f}")

    # 8) MC-dropout predictive mean + variance (std-space)
    mean_mu_s, total_var_s = mc_dropout_predict(
        modules=modules,
        dropout_p=dropout_p,
        X=X_test_s,
        device=device,
        mc_samples=200,
        batch_size=256,
        learn_aleatoric=learn_aleatoric,
    )

    # 9) Un-standardize to original y scale
    mean_mu, total_var = inverse_y(mean_mu_s, total_var_s, y_mu=y_mu, y_sigma=y_sigma)
    total_std = np.sqrt(np.maximum(total_var, 0.0))

    # 10) Print a few examples (first 5 test points, first 3 output dims)
    print("\nExamples (first 5 test samples; first 3 of 48 outputs):")
    for i in range(min(5, mean_mu.shape[0])):
        yt = y_test[i, :3]
        pm = mean_mu[i, :3]
        ps = total_std[i, :3]
        print(f"i={i:02d}  y_true[:3]={yt}  pred_mean[:3]={pm}  pred_std[:3]={ps}")

    print("\nDone.")


if __name__ == "__main__":
    main()
"""
