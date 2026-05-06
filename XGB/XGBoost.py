#!/usr/bin/env python3
# xgb_multioutput_publishable.py
# Publication-ready multi-output regression with XGBoost:
# - Single shared hyperparameter search across all 24 targets (fast, defensible)
# - TimeSeriesSplit CV (leak-safe)
# - Rolling-origin evaluation with window W
# - Robust to older xgboost/sklearn (no fragile kwargs, single-process CV)
# - Self-contained simulation (N=1453, D=248, T=24) for correctness demo

import os, sys, json, math, warnings, time
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42

# ============================== CONFIG (EDIT IF NEEDED) ============================== #
USE_SIMULATED = True  # True → run on synthetic data; False → load CSVs below
X_CSV = r"./X.csv"    # used only if USE_SIMULATED=False
Y_CSV = r"./Y.csv"    # used only if USE_SIMULATED=False

N_ITERS = 5          # RandomizedSearchCV iterations (30–80 is typical)
N_FOLDS = 5           # TimeSeriesSplit folds (>=3)
WINDOW  = 365         # Rolling window length W; if 0/None → auto = min(365, max(30, N//2))
OUT_DIR = "./outputs"
OUT_PREFIX = "xgb_mo"
XGB_THREADS = None     # None → auto = max(1, os.cpu_count()-1)
# ===================================================================================== #

# ------------------------- Library imports (robust) ------------------------- #
try:
    from xgboost import XGBRegressor
except Exception:
    print("XGBoost import failed. Install it:\n  pip install xgboost")
    raise

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import make_scorer

# ------------------------------- Utilities ---------------------------------- #
def set_seeds(seed=RANDOM_STATE):
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass

def threads_default():
    try:
        return max(1, os.cpu_count() - 1)
    except Exception:
        return 1

def detect_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    other = [c for c in df.columns if c not in num_cols + cat_cols]
    return num_cols, cat_cols, other

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols, other = detect_feature_types(X)
    if other:
        print(f"[WARN] Dropping unsupported dtypes: {other}")
    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # dense output for XGB; compatible with old sklearn
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)

def build_xgb(xgb_threads: Optional[int] = None) -> XGBRegressor:
    t = threads_default() if xgb_threads in (None, 0) else int(xgb_threads)
    common = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=RANDOM_STATE,
        seed=RANDOM_STATE,     # very old xgboost compatibility
        n_jobs=t,              # internal threads (CV stays single-process)
    )
    try:  # some old xgboost also reads nthread
        common["nthread"] = t
    except Exception:
        pass
    # Safest constructors first; fallbacks for older builds
    try:
        return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    except TypeError:
        try:
            return XGBRegressor(objective="reg:squarederror", **common)
        except TypeError:
            return XGBRegressor(**common)

def param_distributions() -> Dict[str, List]:
    # We tune the base estimator inside MultiOutputRegressor:
    # Pipeline('pre', 'reg'=MultiOutputRegressor(XGBRegressor)) → 'reg__estimator__<param>'
    return {
        "reg__estimator__n_estimators":    [200, 400, 600, 800, 1200, 1600],
        "reg__estimator__max_depth":       [3, 4, 5, 6, 8, 10],
        "reg__estimator__learning_rate":   [0.3, 0.1, 0.05, 0.03, 0.01],
        "reg__estimator__subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg__estimator__colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
        "reg__estimator__min_child_weight":[1, 2, 3, 5, 7, 10],
        "reg__estimator__gamma":           [0, 0.1, 0.2, 0.5, 1.0],
        "reg__estimator__reg_alpha":       [0.0, 0.001, 0.01, 0.1, 1.0, 10.0],
        "reg__estimator__reg_lambda":      [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }

def rmse_multi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multi-output RMSE across all targets (no fragile kwargs)."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return math.sqrt(float(mse))

def per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    T = y_true.shape[1]
    rows = []
    for j in range(T):
        e = y_true[:, j] - y_pred[:, j]
        mse = float(np.mean(e**2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(e)))
        rows.append({"target_index": j, "rmse": rmse, "mae": mae})
    return pd.DataFrame(rows)

def get_feature_names_safe(pre: ColumnTransformer, n_out: int):
    try:
        names = pre.get_feature_names_out()
        if isinstance(names, np.ndarray):
            names = names.tolist()
        return list(names)
    except Exception:
        return [f"f{i}" for i in range(n_out)]

# ----------------------------- Data loading/sim ----------------------------- #
def load_xy_from_csv(x_path: str, y_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"X_CSV not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Y_CSV not found: {y_path}")
    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)
    if len(X) != len(Y):
        raise ValueError(f"Row mismatch: X has {len(X)}, Y has {len(Y)}.")
    # Force numeric targets; drop any rows with NaN targets
    Y = Y.apply(pd.to_numeric, errors="coerce")
    mask = ~Y.isna().any(axis=1)
    if mask.sum() < len(Y):
        print(f"[WARN] Dropped {int(len(Y)-mask.sum())} rows with NaN targets.")
    return X.loc[mask].reset_index(drop=True), Y.loc[mask].reset_index(drop=True)

def simulate_xy(N=1453, D=248, T=24, seed=RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate realistic 24-dim outputs with daily patterns + regime noise."""
    rng = np.random.RandomState(seed)
    # Features: base Gaussian + weak trends + weekly dummies
    X = rng.randn(N, D)
    # add a slow trend & weekly cycles to 5 random columns
    for j in rng.choice(D, size=5, replace=False):
        X[:, j] += 0.002 * np.arange(N) + 0.5 * np.sin(2*np.pi*np.arange(N)/7.0 + j)
    # Targets: linear map + hourly seasonality + noise + weak auto-reg across days
    W = rng.randn(D, T) * (rng.rand(D, T) < 0.1)  # sparse-ish weights
    base = X @ W
    hours = np.arange(T)[None, :]
    daily_cycle = 5*np.sin(2*np.pi*hours/24.0) + 2*np.cos(4*np.pi*hours/24.0)
    Y = base + daily_cycle  # add shared hour effect
    # add heteroskedastic noise with regime switches
    regime = (np.sin(2*np.pi*np.arange(N)/90.0) > 0).astype(float)  # ~3-month regime
    noise_scale = 0.5 + 0.8*regime[:, None]
    Y = Y + noise_scale * rng.standard_t(df=5, size=(N, T))
    # weak day-lag autocorrelation in targets
    for i in range(1, N):
        Y[i] += 0.2 * (Y[i-1] - base[i-1])
    X_df = pd.DataFrame(X, columns=[f"x{j}" for j in range(D)])
    Y_df = pd.DataFrame(Y, columns=[f"t{h:02d}" for h in range(T)])
    return X_df, Y_df
"""
# ----------------------------- Core Pipeline -------------------------------- #
def main():
    t0 = time.time()
    set_seeds(RANDOM_STATE)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Data
    if USE_SIMULATED:
        X, Y = simulate_xy()
        data_label = "SIMULATED"
    else:
        X, Y = load_xy_from_csv(X_CSV, Y_CSV)
        data_label = "CSV"
    N, D = X.shape
    T = Y.shape[1]
    print(f"\nData [{data_label}] loaded: N={N}, D={D}, T={T}")

    # 2) Window & DEV region for HPO (to avoid peeking)
    if not WINDOW:
        W = min(365, max(30, N // 2))
    else:
        W = int(WINDOW)
    if W >= N:
        raise ValueError(f"WINDOW={W} must be < N={N}.")
    dev_end = max(W, int(0.6 * N))
    print(f"Rolling window W={W} | DEV region for HPO: rows [0:{dev_end}]")

    X_dev, Y_dev = X.iloc[:dev_end].copy(), Y.iloc[:dev_end].copy()

    # 3) Pipeline: preprocessor + MultiOutput(XGB)
    pre = make_preprocessor(X_dev)
    xgb = build_xgb(XGB_THREADS)
    reg = MultiOutputRegressor(xgb, n_jobs=None)  # keep wrapper single-process
    pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])

    # 4) HPO with TimeSeriesSplit and multi-output RMSE scorer
    scorer = make_scorer(rmse_multi, greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=max(3, N_FOLDS))
    dist = param_distributions()
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=dist,
        n_iter=int(N_ITERS),
        scoring=scorer,
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=1,            # Windows stability: single-process CV
        verbose=1,
        error_score=np.nan,
        refit=True           # fit best pipeline on full DEV region
    )
    print("\n[HPO] Searching shared hyperparameters across all targets...")
    rs.fit(X_dev, Y_dev)
    best_pipe = rs.best_estimator_
    best_params = rs.best_params_
    best_cv = float(rs.best_score_)
    print("\n=== Best shared params (TimeSeries CV on DEV) ===")
    for k in sorted(best_params):
        print(f"{k}: {best_params[k]}")
    print(f"Best CV (neg-RMSE): {best_cv:.6f}")

    # Save HPO artifacts
    with open(os.path.join(OUT_DIR, f"{OUT_PREFIX}_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    try:
        pd.DataFrame(rs.cv_results_).to_csv(os.path.join(OUT_DIR, f"{OUT_PREFIX}_cv_results.csv"), index=False)
    except Exception:
        pass

    # Helper to reconstruct pipeline with the best params (fresh estimator each time)
    def make_best_pipeline_from_params(params: Dict) -> Pipeline:
        pre2 = make_preprocessor(X)  # will fit on each train slice
        xgb2 = build_xgb(XGB_THREADS)
        # Apply best params to base XGB
        for key, val in params.items():
            if key.startswith("reg__estimator__"):
                setattr(xgb2, key.split("__")[-1], val)
        reg2 = MultiOutputRegressor(xgb2, n_jobs=None)
        return Pipeline(steps=[("pre", pre2), ("reg", reg2)])

    # 5) Rolling-origin evaluation: predict rows [dev_end, N)
    print(f"\n[ROLLING] Predicting rows [{dev_end}:{N}) with W={W} ...")
    preds, trues, idxs = [], [], []
    for t in range(dev_end, N):
        start = max(0, t - W)
        X_tr, Y_tr = X.iloc[start:t], Y.iloc[start:t]
        X_te = X.iloc[[t]]
        pipe_t = make_best_pipeline_from_params(best_params)
        pipe_t.fit(X_tr, Y_tr)
        yhat_t = pipe_t.predict(X_te)   # (1, T)
        preds.append(yhat_t.reshape(-1))
        trues.append(Y.iloc[t].values)
        idxs.append(t)

    P = np.vstack(preds)    # (n_eval x T)
    Y_true_eval = np.vstack(trues)
    rows_index = pd.Index(idxs, name="row_index")
    pred_cols = [f"pred_t{j}" for j in range(T)]
    true_cols = [f"true_t{j}" for j in range(T)]
    out_df = pd.DataFrame(np.hstack([Y_true_eval, P]), index=rows_index, columns=true_cols + pred_cols)
    out_df.to_csv(os.path.join(OUT_DIR, f"{OUT_PREFIX}_rolling_predictions.csv"), index=True)

    overall_rmse = rmse_multi(Y_true_eval, P)
    pt = per_target_metrics(Y_true_eval, P)
    pt.to_csv(os.path.join(OUT_DIR, f"{OUT_PREFIX}_per_target_metrics.csv"), index=False)

    print("\n=== Rolling-origin metrics (evaluation region) ===")
    print(f"Overall RMSE (all targets): {overall_rmse:.6f}")
    print(f"Per-target metrics → {OUT_DIR}/{OUT_PREFIX}_per_target_metrics.csv")
    print(f"Rolling predictions → {OUT_DIR}/{OUT_PREFIX}_rolling_predictions.csv")

    # 6) Final production model: fit on LAST window -> ready to predict next step
    final_start = max(0, N - W)
    final_pipe = make_best_pipeline_from_params(best_params)
    final_pipe.fit(X.iloc[final_start:N], Y.iloc[final_start:N])

    # Save final pipeline
    try:
        import joblib
        joblib.dump(final_pipe, os.path.join(OUT_DIR, f"{OUT_PREFIX}_final_pipeline.pkl"))
        print(f"Saved final pipeline → {OUT_DIR}/{OUT_PREFIX}_final_pipeline.pkl")
    except Exception:
        print("[WARN] Could not save final pipeline (install joblib if you need it).")

    # Feature importances for target 0 (approx interpretability)
    try:
        pre_step = final_pipe.named_steps["pre"]
        reg_step = final_pipe.named_steps["reg"]
        base_est = reg_step.estimators_[0]
        importances = getattr(base_est, "feature_importances_", None)
        if importances is not None:
            n_out = len(importances)
            feat_names = get_feature_names_safe(pre_step, n_out)
            if len(feat_names) != n_out:
                feat_names = [f"f{i}" for i in range(n_out)]
            fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
            fi.to_csv(os.path.join(OUT_DIR, f"{OUT_PREFIX}_feature_importances_t0.csv"), index=False)
            print(f"Feature importances (target 0) → {OUT_DIR}/{OUT_PREFIX}_feature_importances_t0.csv")
    except Exception:
        pass

    # 7) Run summary (for your paper/repro)
    try:
        import sklearn, xgboost
        summary = {
            "data": {"source": "SIMULATED" if USE_SIMULATED else "CSV", "N": N, "D": D, "T": T},
            "dev_end": dev_end,
            "window_W": W,
            "best_cv_neg_rmse": best_cv,
            "overall_rmse_eval_region": overall_rmse,
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
            "config": {
                "N_ITERS": N_ITERS, "N_FOLDS": N_FOLDS, "WINDOW": WINDOW, "XGB_THREADS": XGB_THREADS
            }
        }
        with open(os.path.join(OUT_DIR, f"{OUT_PREFIX}_run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    print(f"\nDone in {time.time()-t0:.1f}s. Outputs in: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
"""
