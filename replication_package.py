import os
import math
from scipy.stats import norm
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import cvxpy as cp
from pathlib import Path
import argparse
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Needed for Friedman/Nemenyi
import scikit_posthocs as sp


# ======================================================================================
# CONFIG (keep your paths)
# ======================================================================================

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
PRED_DIR = os.path.join(BASE_DIR, "Prediction Result")
TRADING_DIR = os.path.join(BASE_DIR, "Trading_Result")
REGIME_DIR = os.path.join(BASE_DIR, "Regime_Results")

IMAGES_DIR = os.path.join(BASE_DIR, "images")
TABLES_DIR = os.path.join(BASE_DIR, "results_tables")

N_COLS = 24

# Paper uses 7 days (your failure_case_code uses slice(176,183))
SLICE_WEEK = slice(176, 183)

COUNTRIES_2023 = ["germany", "france", "norway"]

# Paper tables/figures use 6 models (no CNP-No-Regimes) except the specific compare table
MODELS_6 = ["R-CNP", "XGB", "DNN", "LEAR", "BLSTM", "TFT"]
MODELS_6_LOWER = ["cnp", "xgb", "dnn", "lear", "blstm", "tft"]

# Paper ordering for TOPSIS table (as in the LaTeX table)
TOPSIS_MODELS = ["R-CNP", "DNN", "LEAR", "XGB", "BLSTM", "TFT"]

MODEL_COLORS_6 = ["blue", "orange", "green", "red", "purple", "brown"]

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: f"{x:.6g}")  # matches paper-style prints


# ======================================================================================
# Matplotlib quality (safe LaTeX toggle)
# ======================================================================================

def _configure_matplotlib_pdf_quality():
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.labelpad": 2,
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
    })

    # fallback if LaTeX is missing
    try:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        fig = plt.figure()
        plt.plot([0, 1], [0, 1])
        tmp_path = os.path.join(IMAGES_DIR, "_latex_test.pdf")
        fig.savefig(tmp_path, bbox_inches="tight")
        plt.close(fig)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    except Exception:
        mpl.rcParams.update({"text.usetex": False})
        try:
            plt.close("all")
        except Exception:
            pass


# ======================================================================================
# IO helpers
# ======================================================================================

def ensure_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

def save_df_csv(df: pd.DataFrame, filename: str):
    df.to_csv(os.path.join(TABLES_DIR, filename), index=True)

def load_txt(path: str, ndmin: int = 1) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file:\n  {path}")
    arr = np.loadtxt(path)
    if ndmin == 2 and arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def load_matrix_2d(path: str, n_cols: int = N_COLS, max_rows: Optional[int] = 365) -> np.ndarray:
    arr = load_txt(path, ndmin=2)
    if arr.shape[1] < n_cols:
        raise ValueError(f"File has {arr.shape[1]} columns (<{n_cols}):\n  {path}")
    arr = arr[:, :n_cols]
    if max_rows is not None:
        arr = arr[:max_rows, :]
    return arr

def load_vector(path: str, max_rows: Optional[int] = None) -> np.ndarray:
    arr = load_txt(path, ndmin=1).astype(float).ravel()
    if max_rows is not None:
        arr = arr[:max_rows]
    return arr

#======================================================================================
# sesnsitivity table helpers
#======================================================================================

# ===========================
# Sensitivity (Case-III) helpers
# ===========================

T_HORIZON = 24
C_max = 0.01
x_max_net = 0.003
x_min_net = -0.003
eta_c = 0.95
eta_d = 0.95
s_init = 0.0
s_final = 0.0

lambda_risk = 0.5
mu_residual = 1.11
gamma_renewable = 1.67

P_charge_max = abs(x_min_net)
P_discharge_max = x_max_net


def remove_nan(data_set: np.ndarray) -> np.ndarray:
    data_set = np.array(data_set, dtype=float, copy=True)
    for i in range(data_set.shape[0]):
        prc_row = data_set[i, :]

        if np.count_nonzero(np.isnan(prc_row)) == 24 and i == 0:
            prc_row = (data_set[i + 1, :] + data_set[i + 2, :]) / 2
        elif np.count_nonzero(np.isnan(prc_row)) == 24 and 0 < i < (data_set.shape[0] - 1):
            prc_row = (data_set[i - 1, :] + data_set[i + 1, :]) / 2
        elif np.count_nonzero(np.isnan(prc_row)) == 24 and i == (data_set.shape[0] - 1):
            prc_row = (data_set[i - 1, :] + data_set[i - 2, :]) / 2
        elif np.count_nonzero(np.isnan(prc_row)) == 24 and i == (data_set.shape[0] - 2):
            prc_row = (data_set[i + 1, :] + data_set[i + 2, :]) / 2
        elif 0 < np.count_nonzero(np.isnan(prc_row)) == 23:
            prc_row = (data_set[i - 1, :] + data_set[i + 1, :]) / 2
        elif 0 < np.count_nonzero(np.isnan(prc_row)) < 24 and np.argwhere(np.isnan(prc_row))[0][0] != 23:
            j = int(np.argwhere(np.isnan(prc_row))[0][0])
            prc_row[j] = (prc_row[j - 1] + prc_row[j + 1]) / 2
        elif 0 < np.count_nonzero(np.isnan(prc_row)) < 24 and np.argwhere(np.isnan(prc_row))[0][0] == 0:
            j = int(np.argwhere(np.isnan(prc_row))[0][0])
            prc_row[j] = (prc_row[j + 1] + prc_row[j + 2]) / 2
        elif 0 < np.count_nonzero(np.isnan(prc_row)) < 24 and np.argwhere(np.isnan(prc_row))[0][0] == 23:
            j = int(np.argwhere(np.isnan(prc_row))[0][0])
            prc_row[j] = (prc_row[j - 1] + prc_row[j - 2]) / 2

        data_set[i, :] = prc_row
    return data_set


def solve_case_iii(prices, realized_prices, realized_load, realized_renewable_availability,
                   uncertainty, load, renewable_availability, is_perfect_foresight=False):
    """
    Matches your snippet:
    - If is_perfect_foresight: no uncertainty penalty and uses 'prices' as realized prices.
    - Otherwise: includes uncertainty penalty in planning objective
    - Returns realized-profit objective computed using realized exogenous variables.
    """

    x_plus = cp.Variable(T_HORIZON, nonneg=True)
    x_minus = cp.Variable(T_HORIZON, nonneg=True)
    s = cp.Variable(T_HORIZON + 1)

    prices = np.asarray(prices, dtype=float).ravel()
    realized_prices = np.asarray(realized_prices, dtype=float).ravel()
    realized_load = np.asarray(realized_load, dtype=float).ravel()
    realized_renewable_availability = np.asarray(realized_renewable_availability, dtype=float).ravel()
    uncertainty = np.asarray(uncertainty, dtype=float).ravel()
    load = np.asarray(load, dtype=float).ravel()
    renewable_availability = np.asarray(renewable_availability, dtype=float).ravel()

    if is_perfect_foresight:
        objective = cp.Maximize(
            cp.sum(prices @ (x_minus * eta_d - x_plus / eta_c)
                   - mu_residual * load @ x_plus
                   + gamma_renewable * renewable_availability @ x_plus)
        )
    else:
        objective = cp.Maximize(
            cp.sum(prices @ (x_minus * eta_d - x_plus / eta_c)
                   - lambda_risk * uncertainty @ (x_plus + x_minus)
                   - mu_residual * load @ x_plus
                   + gamma_renewable * renewable_availability @ x_plus)
        )

    constraints = [
        s[0] == s_init,
        s[1:] == s[:-1] + eta_c * x_plus - (1 / eta_d) * x_minus,
        s >= 0,
        s <= C_max,
        s[T_HORIZON] >= s_final,
        x_plus <= P_charge_max,
        x_minus <= P_discharge_max,
        x_min_net <= x_minus - x_plus,
        x_minus - x_plus <= x_max_net
    ]

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            actual_profit = (
                realized_prices @ (x_minus.value * eta_d - x_plus.value / eta_c)
                - mu_residual * (realized_load @ x_plus.value)
                + gamma_renewable * (realized_renewable_availability @ x_plus.value)
            )
            return float(actual_profit), x_plus.value, x_minus.value, s.value
        return None, None, None, None
    except Exception:
        return None, None, None, None


# ======================================================================================
# Metrics
# ======================================================================================

def mae(pred: np.ndarray, real: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.mean(np.abs(pred - real), axis=axis)

def rmse(pred: np.ndarray, real: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.sqrt(np.mean((pred - real) ** 2, axis=axis))

def smape(pred: np.ndarray, real: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    num = np.abs(pred - real)
    den = np.abs(real) + np.abs(pred) + eps
    return 200.0 * np.mean(num / den, axis=axis)

def compute_uncertainty_metrics(real: np.ndarray, low: np.ndarray, high: np.ndarray) -> Tuple[float, float]:
    inside = (real >= low) & (real <= high)
    picp = float(np.mean(inside))
    mpiw = float(np.mean(high - low))
    return picp, mpiw


# ======================================================================================
# DM test (h=1, MAE loss) – matches your usage
# ======================================================================================

def _try_scipy_t_cdf(x: float, df: int) -> Optional[float]:
    try:
        from scipy.stats import t as student_t
        return float(student_t.cdf(x, df=df))
    except Exception:
        return None

def diebold_mariano_test(actual, forecast1, forecast2, loss_function='mse', h=1):
    """
    Computes the Diebold-Mariano test statistic and p-value for comparing the predictive accuracy of two forecasts.

    Args:
        actual (array-like): Array of actual values.
        forecast1 (array-like): Array of predictions from model 1.
        forecast2 (array-like): Array of predictions from model 2.
        loss_function (str, optional):  The loss function to use.  Options are:
            'mse' (Mean Squared Error, default)
            'rmse' (Root Mean Squared Error)
            'mae' (Mean Absolute Error)
            'mase' (Mean Absolute Scaled Error - requires a training set of actuals)
            'percentage' (Absolute Percentage Error)
            'log' (Logarithmic loss, good for probability forecasts).
            You can also pass a *function* that accepts (actual, forecast)
            and returns a loss.
        h (int, optional): The forecast horizon.  Defaults to 1.

    Returns:
        tuple: (DM statistic, p-value).  Returns (np.nan, np.nan) if there are errors.
    """

    # 1. Input Validation and Error Handling
    try:
        actual = np.asarray(actual)
        forecast1 = np.asarray(forecast1)
        forecast2 = np.asarray(forecast2)
    except Exception as e:
        warnings.warn(f"Error converting inputs to numpy arrays: {e}")
        return np.nan, np.nan

    if not all(x.shape == actual.shape for x in [forecast1, forecast2]):
        warnings.warn(" forecast arrays must have the same shape as the actual array.")
        return np.nan, np.nan

    if actual.size < 2:
        warnings.warn("Insufficient data:  Need at least two data points.")
        return np.nan, np.nan

    if h < 1:
        warnings.warn("Horizon 'h' must be >= 1")
        return np.nan, np.nan

    # 2. Define and Calculate Loss Differential
    def mse(actual, forecast):
        return (actual - forecast) ** 2

    def rmse(actual, forecast):
        return np.sqrt(mse(actual, forecast))
    def mae(actual, forecast):
        return np.abs(actual - forecast)

    def mase(actual, forecast, actual_train=None, forecast_train=None):
        """
        Mean Absolute Scaled Error.  Requires a training set to calculate the
        scaling factor.
        """
        if actual_train is None or forecast_train is None:
            raise ValueError("MASE requires actual_train and forecast_train")

        if len(actual_train) < 2:
             raise ValueError("MASE requires at least two training data points.")
        scale = np.mean(np.abs(actual_train[1:] - actual_train[:-1]))
        if scale == 0:
            return np.nan  # Handle the case where the training data is constant
        return np.mean(np.abs(actual - forecast)) / scale

    def percentage_error(actual, forecast):
        """
        Absolute Percentage Error (APE).  Note: Can be unstable if actual
        values are close to zero.  Returns values in range [0, 100].
        """
        return np.abs((actual - forecast) / actual) * 100
    
    def log_loss(actual, forecast):
        """
        Logarithmic loss.  Useful for probability forecasts.  Assumes
        actual values are 0 or 1, and forecasts are probabilities.
        """
        # Clip forecasts to avoid log(0) and log(1) errors.
        forecast_clipped = np.clip(forecast, 1e-15, 1 - 1e-15)
        return - (actual * np.log(forecast_clipped) + (1 - actual) * np.log(1 - forecast_clipped))

    # Select the loss function
    if callable(loss_function):
        loss_func = loss_function  # Use the user-provided function directly
    elif isinstance(loss_function, str):
        loss_function = loss_function.lower()
        if loss_function == 'mse':
            loss_func = mse
        elif loss_function == 'rmse':
            loss_func = rmse
        elif loss_function == 'mae':
            loss_func = mae
        elif loss_function == 'mase':
            #The check for actual_train and forecast_train is now inside mase
            loss_func = mase
        elif loss_function == 'percentage':
            loss_func = percentage_error
        elif loss_function == 'log':
            loss_func = log_loss
        else:
            warnings.warn(f"Invalid loss function: {loss_function}.  Using MSE.")
            loss_func = mse
    else:
        warnings.warn(f"Invalid loss function type: {type(loss_function)}. Using MSE.")
        loss_func = mse

    # Calculate the loss differential
    if loss_function == 'mase':
      #Need to pass the training data.  Assume first half is training, second half is testing
      train_size = actual.size // 2
      loss_diff = loss_func(actual, forecast1, actual[:train_size], forecast1[:train_size]) - loss_func(actual, forecast2, actual[:train_size], forecast2[:train_size])
    else:
       loss_diff = loss_func(actual, forecast1) - loss_func(actual, forecast2)

    # 3. Compute the DM Statistic
    d_mean = np.mean(loss_diff)
    if d_mean == 0:
        warnings.warn("Loss differential has zero mean.  DM statistic is undefined.")
        return np.nan, np.nan

    # Autocovariance calculation using the Bartlet kernel (h-1 lags)
    n = len(loss_diff)
    gamma_sum = 0
    for k in range(1, h):
        gamma = np.sum((loss_diff[k:] - d_mean) * (loss_diff[:-k] - d_mean)) / n
        gamma_sum += 2 * (1 - k / h) * gamma  # Bartlet kernel weight

    # Handle the case where the variance is zero.  This can happen
    # if the loss differential is constant.
    var_loss_diff = np.var(loss_diff)
    if var_loss_diff == 0:
        warnings.warn("Loss differential has zero variance.  DM statistic is undefined.")
        return np.nan, np.nan

    #DM statistic
    dm_statistic = d_mean / np.sqrt((var_loss_diff + gamma_sum) / n)

    # 4. Calculate the p-value
    p_value = 2 * (1 - norm.cdf(np.abs(dm_statistic)))

    return dm_statistic, p_value


# ======================================================================================
# Alignment rule (paper): Germany-2021 + DNN uses real[3:]
# ======================================================================================

def align_real_pred(country: str, year: int, model: str,
                    real: np.ndarray, pred: np.ndarray, var: Optional[np.ndarray] = None
                    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    real2, pred2, var2 = real, pred, var
    if (country.lower() == "germany") and (year == 2021) and (model.upper() == "DNN"):
        real2 = real2[3:, :]
    T = min(real2.shape[0], pred2.shape[0])
    real2 = real2[:T, :]
    pred2 = pred2[:T, :]
    if var2 is not None:
        var2 = var2[:T, :]
    return real2, pred2, var2


# ======================================================================================
# Paths (keep your naming patterns)
# ======================================================================================

def path_real(country: str, year: int) -> str:
    c = country.lower()
    if year == 2023 and c in ("germany", "france", "norway"):
        return os.path.join(PRED_DIR, "real_data", f"real_{c}.txt")
    return os.path.join(PRED_DIR, "real_data", f"real_{c}_{year}.txt")

def path_pred_var(country: str, year: int, model: str) -> Tuple[str, str]:
    c = country.lower()
    m = model.upper()

    if year == 2023:
        if m == "R-CNP":
            pred = os.path.join(PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned",
                                f"Result_{c}_tuned_new", f"pred_cnp_{c}.txt")
            var = os.path.join(PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned",
                               f"Result_{c}_tuned_new", f"var_cnp_{c}.txt")
            return pred, var
        if m == "XGB":
            pred = os.path.join(PRED_DIR, "XGB", f"pred_xbg_{c}_2023.txt")  # xbg in your files
            var = os.path.join(PRED_DIR, "XGB", f"var_xgb_{c}.txt")
            return pred, var
        if m == "DNN":
            pred = os.path.join(PRED_DIR, "DNN", f"DNN_{c}_2023.txt")
            var = os.path.join(PRED_DIR, "DNN", f"var_dnn_{c}.txt")
            return pred, var
        if m == "LEAR":
            pred = os.path.join(PRED_DIR, "LEAR", f"Final_pred_LEAR_{c}.txt")
            var = os.path.join(PRED_DIR, "LEAR", f"var_lear_{c}.txt")
            return pred, var
        if m == "BLSTM":
            pred = os.path.join(PRED_DIR, "BLSTM", f"pred_blstm_{c}.txt")
            var = os.path.join(PRED_DIR, "BLSTM", f"var_blstm_{c}.txt")
            return pred, var
        if m == "TFT":
            pred = os.path.join(PRED_DIR, "TFT", f"pred_tft_{c}.txt")
            var = os.path.join(PRED_DIR, "TFT", f"var_tft_{c}.txt")
            return pred, var
        if m == "CNP-NO-REGIMES":
            pred = os.path.join(PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned", "CNP_no_regimes",
                                f"pred_{c}_no_regimes_new.txt")
            var = os.path.join(PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned", "CNP_no_regimes",
                               f"var_{c}_no_regimes_new.txt")
            return pred, var

    if c == "germany" and year in (2021, 2022):
        if m == "R-CNP":
            pred = os.path.join(PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned",
                                f"Result_germany_tuned_{year}", f"pred_cnp_germany_{year}.txt")
            var = os.path.join(PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned",
                               f"Result_germany_tuned_{year}", f"var_cnp_germany_{year}.txt")
            return pred, var
        if m == "XGB":
            pred = os.path.join(PRED_DIR, "XGB", f"pred_xgb_germany_{year}.txt")
            var = os.path.join(PRED_DIR, "XGB", f"var_xgb_{year}_germany.txt")
            return pred, var
        if m == "DNN":
            pred = os.path.join(PRED_DIR, "DNN", f"DNN_germany_{year}.txt")
            var = os.path.join(PRED_DIR, "DNN", f"var_dnn_{year}_germany.txt")
            return pred, var
        if m == "LEAR":
            pred = os.path.join(PRED_DIR, "LEAR", f"Final_pred_LEAR_Germany_{year}.txt")
            var = os.path.join(PRED_DIR, "LEAR", f"var_lear_{year}_germany.txt")
            return pred, var
        if m == "BLSTM":
            pred = os.path.join(PRED_DIR, "BLSTM", f"pred_blstm_germany_{year}.txt")
            var = os.path.join(PRED_DIR, "BLSTM", f"var_blstm_germany_{year}.txt")
            return pred, var
        if m == "TFT":
            pred = os.path.join(PRED_DIR, "TFT", f"pred_tft_germany_{year}.txt")
            var = os.path.join(PRED_DIR, "TFT", f"var_tft_{year}_germany.txt")
            return pred, var
        if m == "CNP-NO-REGIMES":
            pred = os.path.join(PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned", "CNP_no_regimes",
                                f"pred_germany_no_regimes_{year}.txt")
            var = os.path.join(PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned", "CNP_no_regimes",
                               f"var_germany_no_regimes_{year}_new.txt")
            return pred, var

    raise ValueError(f"Unsupported (country, year, model): {country}, {year}, {model}")


# ======================================================================================
# Trading paths
# ======================================================================================

def _trading_subdir_for_model(model: str) -> str:
    m = model.lower()
    if m == "r-cnp":
        return "cnp_trading"
    return f"{m}_trading"

def path_trading_series(country: str, year: int, model: str, case: str, kind: str) -> str:
    c = country.lower()
    sub = _trading_subdir_for_model(model)
    base = os.path.join(TRADING_DIR, f"Trading_Result_{c}", sub)

    mtag = "cnp" if model.upper() == "R-CNP" else model.lower()

    if year == 2023:
        if kind == "pf":
            return os.path.join(base, f"pf_{mtag}_case_{case}.txt")
        if kind == "profit":
            return os.path.join(base, f"profit_{mtag}_case_{case}.txt")
        if kind == "cost":
            return os.path.join(base, f"cost_{mtag}_case_{case}.txt")

    if c == "germany" and year in (2021, 2022):
        if kind == "pf":
            return os.path.join(base, f"pf_{mtag}_{year}_case_{case}.txt")
        if kind == "profit":
            return os.path.join(base, f"profit_{mtag}_{year}_case_{case}.txt")
        if kind == "cost":
            return os.path.join(base, f"cost_{mtag}_{year}_case_{case}.txt")

    raise ValueError(f"Unsupported trading path: {country}, {year}, {model}, case={case}, kind={kind}")


# ======================================================================================
# Plots (match paper filenames)
# ======================================================================================

def plot_error_curves_rmse(country: str, year: int, save_path: str):
    real = load_matrix_2d(path_real(country, year), n_cols=N_COLS)
    curves = []
    for m in MODELS_6:
        pred_path, _ = path_pred_var(country, year, m)
        pred = load_matrix_2d(pred_path, n_cols=N_COLS)
        real_m, pred_m, _ = align_real_pred(country, year, m, real, pred, None)
        curves.append(rmse(pred_m, real_m, axis=1))

    T = min(len(c) for c in curves)
    curves = [c[:T] for c in curves]
    x = np.arange(T)  # start at 0 to match paper tick style

    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 5))
    axes = axes.ravel()
    ymin = min(c.min() for c in curves)
    ymax = max(c.max() for c in curves)

    for ax, c, lab, col in zip(axes, curves, MODELS_6, MODEL_COLORS_6):
        ax.plot(x, c, color=col, linewidth=1.6)
        ax.set_title(lab)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ymin, ymax)

    fig.supxlabel(f"Days of {year}")
    fig.supylabel("RMSE")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    fig.savefig(save_path, bbox_inches="tight")
    #plt.close(fig)

def plot_models_comparison_week(country: str, year: int, save_path: str, week_slice=slice(176,182)):
    real = load_matrix_2d(path_real(country, year), n_cols=N_COLS)
    preds, vars_ = [], []
    for m in MODELS_6:
        pred_path, var_path = path_pred_var(country, year, m)
        pred = load_matrix_2d(pred_path, n_cols=N_COLS)
        var = load_matrix_2d(var_path, n_cols=N_COLS)
        real_m, pred_m, var_m = align_real_pred(country, year, m, real, pred, var)
        preds.append(pred_m)
        vars_.append(var_m)

    sl = week_slice
    real_w = real[sl].ravel()
    preds_w = [p[sl].ravel() for p in preds]
    vars_w = [v[sl].ravel() for v in vars_]

    T = min([len(real_w)] + [len(p) for p in preds_w] + [len(v) for v in vars_w])
    x = np.arange(T)

    real_w = real_w[:T]
    preds_w = [p[:T] for p in preds_w]
    vars_w = [v[:T] for v in vars_w]

    z = 1.96
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 5))
    axes = axes.ravel()

    all_lows, all_highs = [], []
    for p, v in zip(preds_w, vars_w):
        std = np.sqrt(np.maximum(v, 0.0))
        all_lows.append((p - z * std).min())
        all_highs.append((p + z * std).max())
    ymin = min(real_w.min(), min(all_lows))
    ymax = max(real_w.max(), max(all_highs))

    for ax, p, v, lab, col in zip(axes, preds_w, vars_w, MODELS_6, MODEL_COLORS_6):
        std = np.sqrt(np.maximum(v, 0.0))
        lo = p - z * std
        hi = p + z * std
        ax.plot(x, real_w, color="black", linestyle="--", linewidth=1.2, label="Real")
        ax.plot(x, p, color=col, linewidth=1.6, label=lab)
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, linewidth=0)
        ax.set_title(lab)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ymin, ymax)

    fig.supxlabel(f"Days of {year}")
    fig.supylabel("Price")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.25, hspace=0.35)
    fig.savefig(save_path, bbox_inches="tight")
    #plt.close(fig)

def make_rgb_pval_cmap():
    red = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], [0]])
    green = np.concatenate([np.linspace(0.5, 1, 50), np.zeros(50)])
    blue = np.zeros(100)
    rgb = np.concatenate([red.reshape(-1, 1), green.reshape(-1, 1), blue.reshape(-1, 1)], axis=1)
    return mpl.colors.ListedColormap(rgb)

def dm_pval_heatmap(ax, P, labels, title, cmap, vmin=0.0, vmax=0.1):
    P = np.asarray(P, dtype=float)
    im = ax.imshow(P, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    n = P.shape[0]
    ax.plot(np.arange(n), np.arange(n), "wx", markersize=6, markeredgewidth=1.2)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.grid(False)
    return im

def plot_dm_panels_germany(dm_pvals_by_year: Dict[int, np.ndarray], save_path: str):
    cmap = make_rgb_pval_cmap()
    titles = {2021: r"2021", 2022: r"2022", 2023: r"2023"}

    fig = plt.figure(figsize=(12.5, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.40)

    axes = []
    im_last = None
    for i, year in enumerate([2021, 2022, 2023]):
        ax = fig.add_subplot(gs[0, i], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
        im_last = dm_pval_heatmap(ax, dm_pvals_by_year[year], MODELS_6, titles[year], cmap, vmin=0.0, vmax=0.1)
        axes.append(ax)
        if i != 0:
            ax.tick_params(labelleft=False)

    cax = fig.add_subplot(gs[0, 3])
    fig.colorbar(im_last, cax=cax)
    fig.savefig(save_path, bbox_inches="tight")
    #plt.close(fig)

def plot_regime_plots_2023(country: str):
    c = country.lower()
    regime_length_path = os.path.join(REGIME_DIR, f"Regime_Results_{c}", f"regime_length_{c}_2023.txt")
    ids_path = os.path.join(REGIME_DIR, f"Regime_Results_{c}", f"regime_train_output_{c}.txt")
    train_output_path = os.path.join(REGIME_DIR, f"Regime_Results_{c}", f"train_output_{c}.txt")

    regime_length = load_vector(regime_length_path)
    ids = load_vector(ids_path)
    train_output = load_vector(train_output_path)

    fig = plt.figure(figsize=(5.5, 5.5))
    plt.plot(regime_length)
    plt.axhline(y=3, color='r', linestyle='--')
    plt.axhline(y=6, color='g', linestyle='--')
    plt.xlabel('Days from Jan, 2023 to Dec, 2023')
    plt.ylabel('Number of Regimes')
    fig.savefig(os.path.join(IMAGES_DIR, f"regime_length_{c}.pdf"), bbox_inches="tight")
    #plt.close(fig)


    unique_id = np.unique(ids)
    id_labels = [f'Regime {i}' for i in unique_id]
    custom_first_colors = ['blue', 'orange', 'green', 'red']
    all_colors = []
    for i in range(min(len(unique_id), len(custom_first_colors))):
        all_colors.append(custom_first_colors[i])
    if len(unique_id) > len(custom_first_colors):
        base_cmap = cm.get_cmap('viridis', len(unique_id) - len(custom_first_colors))
        for i in range(len(unique_id) - len(custom_first_colors)):
            all_colors.append(base_cmap(i))
    cmap = ListedColormap(all_colors)
    bounds = np.concatenate([unique_id - 0.5, [unique_id[-1] + 0.5]])
    norm = BoundaryNorm(bounds, cmap.N)
    fig = plt.figure(figsize=(5, 5))
    scatter = plt.scatter(np.arange(len(train_output)), train_output, c=ids, cmap=cmap,norm=norm, s=10)
    plt.xlabel('Days from Training Data', fontweight = 'bold',fontsize=12)
    plt.ylabel('Scaled Hourly Averaged Price in Euro/MWh', fontweight = 'bold',fontsize=12)
    fig.savefig(os.path.join(IMAGES_DIR, f"regime_detect_{c}.pdf"), bbox_inches="tight")
    #plt.close(fig)


# ======================================================================================
# Tables (match paper structure)
# ======================================================================================

SCENARIOS = [
    ("germany", 2021, "Germany-2021"),
    ("germany", 2022, "Germany-2022"),
    ("germany", 2023, "Germany-2023"),
    ("france",  2023, "France-2023"),
    ("norway",  2023, "Norway-2023"),
]

def scenario_columns():
    return [lab for _, _, lab in SCENARIOS]

def compute_error_summary(country: str, year: int, model: str) -> Tuple[float, float, float]:
    real = load_matrix_2d(path_real(country, year), n_cols=N_COLS)
    pred_path, _ = path_pred_var(country, year, model)
    pred = load_matrix_2d(pred_path, n_cols=N_COLS)
    real_m, pred_m, _ = align_real_pred(country, year, model, real, pred, None)
    return (
        float(np.mean(mae(pred_m, real_m, axis=1))),
        float(np.mean(rmse(pred_m, real_m, axis=1))),
        float(np.mean(smape(pred_m, real_m, axis=1))),
    )

def compute_picp_mpiw_summary(country: str, year: int, model: str) -> Tuple[float, float]:
    real = load_matrix_2d(path_real(country, year), n_cols=N_COLS)
    pred_path, var_path = path_pred_var(country, year, model)
    pred = load_matrix_2d(pred_path, n_cols=N_COLS)
    var = load_matrix_2d(var_path, n_cols=N_COLS)
    real_m, pred_m, var_m = align_real_pred(country, year, model, real, pred, var)
    std = np.sqrt(np.maximum(var_m, 0.0))
    lo = pred_m - 1.96 * std
    hi = pred_m + 1.96 * std
    return compute_uncertainty_metrics(real_m, lo, hi)

def table_sensitivity_hyperparams_compact() -> pd.DataFrame:
    # ---- sensitivity cases (folders) ----
    sensetivity_cases = [
        "case-I","case-II","case-III","case-IV","case-V","case-VI",
        "case-VII","case-VIII","case-IX","case-X","case-XI"
    ]

    # ---- config labels (manual) ----
    CONFIGS = [
        dict(name="BASE",
             alpha0_a_pri=2.0, alpha0_b_pri=1.0, gamma0_a_pri=3.0, gamma0_b_pri=1.0,
             v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),
        dict(name="ALPHA_LOW",
             alpha0_a_pri=1.0, alpha0_b_pri=1.0, gamma0_a_pri=3.0, gamma0_b_pri=1.0,
             v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),
        dict(name="ALPHA_HIGH",
             alpha0_a_pri=4.0, alpha0_b_pri=1.0, gamma0_a_pri=3.0, gamma0_b_pri=1.0,
             v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),
        dict(name="GAMMA_LOW",
             alpha0_a_pri=2.0, alpha0_b_pri=1.0, gamma0_a_pri=1.5, gamma0_b_pri=1.0,
             v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),
        dict(name="GAMMA_HIGH",
             alpha0_a_pri=2.0, alpha0_b_pri=1.0, gamma0_a_pri=6.0, gamma0_b_pri=1.0,
             v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),
        dict(name="STICK_LOW",
             alpha0_a_pri=2.0, alpha0_b_pri=1.0, gamma0_a_pri=3.0, gamma0_b_pri=1.0,
             v0_range=(0.70, 0.72), v1_range=(0.44, 0.48), p=3),
        dict(name="STICK_HIGH",
             alpha0_a_pri=2.0, alpha0_b_pri=1.0, gamma0_a_pri=3.0, gamma0_b_pri=1.0,
             v0_range=(0.94, 0.96), v1_range=(0.28, 0.31), p=3),
        dict(name="LOW_LOW",
             alpha0_a_pri=1.0, alpha0_b_pri=1.0, gamma0_a_pri=1.5, gamma0_b_pri=1.0,
             v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),
        dict(name="HIGH_HIGH",
             alpha0_a_pri=4.0, alpha0_b_pri=1.0, gamma0_a_pri=6.0, gamma0_b_pri=1.0,
             v0_range=(0.84, 0.86), v1_range=(0.35, 0.39), p=3),
        dict(name="ALPHA_HIGH_STICK_HIGH",
             alpha0_a_pri=4.0, alpha0_b_pri=1.0, gamma0_a_pri=3.0, gamma0_b_pri=1.0,
             v0_range=(0.94, 0.96), v1_range=(0.28, 0.31), p=3),
        dict(name="GAMMA_HIGH_STICK_LOW",
             alpha0_a_pri=2.0, alpha0_b_pri=1.0, gamma0_a_pri=6.0, gamma0_b_pri=1.0,
             v0_range=(0.70, 0.72), v1_range=(0.44, 0.48), p=3),
    ]

    if len(CONFIGS) != len(sensetivity_cases):
        raise ValueError("CONFIGS and sensetivity_cases must have the same length/order.")

    # ---- load shared sensitivity data ----
    real_path = os.path.join(PRED_DIR, "real_data", "real_sensetivity.txt")
    real_full = load_matrix_2d(real_path, n_cols=24)

    load_path = os.path.join(BASE_DIR, "Trading_Strategies", "load_sensetivity.txt")
    pv_path = os.path.join(BASE_DIR, "Trading_Strategies", "pv_sensetivity.txt")
    load_full = load_matrix_2d(load_path, n_cols=24)
    pv_full = load_matrix_2d(pv_path, n_cols=24)

    # Your profit snippet uses first 179 days
    max_days_profit = 179

    rows = []
    for cs, cfg in zip(sensetivity_cases, CONFIGS):
        # prediction/variance files
        pred_path = os.path.join(
            PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned", "Result_germany_sensetivity",
            f"{cs}_new", f"pred_cnp_{cs}.txt"
        )
        var_path = os.path.join(
            PRED_DIR, "CNP", "Result_CNP_HYPER_Tuned", "Result_germany_sensetivity",
            f"{cs}_new", f"var_cnp_{cs}.txt"
        )

        pred_full = load_matrix_2d(pred_path, n_cols=24)
        var_full = load_matrix_2d(var_path, n_cols=24)

        # ---------- MAE (matches your snippet logic) ----------
        # You did: real - pred_cases[:-1,:24]
        # Handle both "pred has extra last row" and "same length" safely.
        if pred_full.shape[0] == real_full.shape[0] + 1:
            pred_mae = pred_full[:-1, :24]
            real_mae = real_full[:, :24]
        else:
            Tm = min(pred_full.shape[0], real_full.shape[0])
            pred_mae = pred_full[:Tm, :24]
            real_mae = real_full[:Tm, :24]

        mae_val = float(np.mean(np.mean(np.abs(real_mae - pred_mae), axis=1)))

        # ---------- Profit (Case-III CVXPY, matches your loop) ----------
        Tp = min(max_days_profit, pred_full.shape[0], var_full.shape[0], real_full.shape[0], load_full.shape[0], pv_full.shape[0])
        pred_p = remove_nan(pred_full[:Tp, :24])
        var_p = remove_nan(var_full[:Tp, :24])
        real_p = remove_nan(real_full[:Tp, :24])
        load_p = remove_nan(load_full[:Tp, :24])
        pv_p = remove_nan(pv_full[:Tp, :24])

        profits = np.full(Tp, np.nan, dtype=float)

        for d in range(Tp):
            realized_prices = real_p[d]
            realized_load = load_p[d]
            realized_pv = pv_p[d]

            prof, _, _, _ = solve_case_iii(
                pred_p[d],
                realized_prices,
                realized_load,
                realized_pv,
                np.sqrt(np.maximum(var_p[d], 0.0)),
                load_p[d],
                pv_p[d],
                is_perfect_foresight=False
            )
            if prof is not None:
                profits[d] = prof

        profit_mean = float(np.nanmean(profits))

        rows.append({
            "Config": cfg["name"],
            "MAE": mae_val,
            "Profit": profit_mean
        })

    df = pd.DataFrame(rows).set_index("Config")[["MAE", "Profit"]]
    return df

def table_rcnp_vs_noregimes_all() -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([scenario_columns(), ["MAE", "RMSE"]])
    df = pd.DataFrame(index=["R-CNP", "CNP-No-Regimes"], columns=cols, dtype=float)

    for c, y, lab in SCENARIOS:
        for model in ["R-CNP", "CNP-NO-REGIMES"]:
            real = load_matrix_2d(path_real(c, y), n_cols=N_COLS)
            pred_path, _ = path_pred_var(c, y, model)
            pred = load_matrix_2d(pred_path, n_cols=N_COLS)
            real_m, pred_m, _ = align_real_pred(c, y, model, real, pred, None)
            df.loc["R-CNP" if model == "R-CNP" else "CNP-No-Regimes", (lab, "MAE")] = float(np.mean(mae(pred_m, real_m, axis=1)))
            df.loc["R-CNP" if model == "R-CNP" else "CNP-No-Regimes", (lab, "RMSE")] = float(np.mean(rmse(pred_m, real_m, axis=1)))

    return df

def table_all_errors_merged_all() -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([scenario_columns(), ["MAE", "RMSE", "SMAPE"]])
    df = pd.DataFrame(index=MODELS_6, columns=cols, dtype=float)

    for c, y, lab in SCENARIOS:
        for model in MODELS_6:
            a, b, s = compute_error_summary(c, y, model)
            df.loc[model, (lab, "MAE")] = a
            df.loc[model, (lab, "RMSE")] = b
            df.loc[model, (lab, "SMAPE")] = s

    return df

def table_picp_mpiw_merged_all() -> pd.DataFrame:
    cols = pd.MultiIndex.from_product([scenario_columns(), ["PICP", "MPIW"]])
    df = pd.DataFrame(index=MODELS_6, columns=cols, dtype=float)

    for c, y, lab in SCENARIOS:
        for model in MODELS_6:
            picp, mpiw = compute_picp_mpiw_summary(c, y, model)
            df.loc[model, (lab, "PICP")] = picp
            df.loc[model, (lab, "MPIW")] = mpiw

    return df

def failure_case_extremes_table_france_2023() -> pd.DataFrame:
    # Matches paper tab:7day_extremes content
    country, year = "france", 2023
    sl = SLICE_WEEK

    real = load_matrix_2d(path_real(country, year), n_cols=N_COLS)
    real_w = real[sl, :]

    # Load preds/vars for 6 models
    preds = {}
    vars_ = {}
    for m in MODELS_6:
        pred_path, var_path = path_pred_var(country, year, m)
        pred = load_matrix_2d(pred_path, n_cols=N_COLS)
        var = load_matrix_2d(var_path, n_cols=N_COLS)
        real_m, pred_m, var_m = align_real_pred(country, year, m, real, pred, var)
        preds[m] = pred_m[sl, :]
        vars_[m] = var_m[sl, :]

    days = real_w.shape[0]  # should be 7
    idx = [f"Day{i+1}" for i in range(days)]

    def coverage_at_real_extreme(day_row: np.ndarray, pred_row: np.ndarray, var_row: np.ndarray, which: str) -> str:
        # hour at which REAL hits max/min, check if that real value is inside model PI at that hour
        if which == "max":
            h = int(np.argmax(day_row))
        else:
            h = int(np.argmin(day_row))
        real_val = float(day_row[h])
        std = math.sqrt(max(float(var_row[h]), 0.0))
        lo = float(pred_row[h] - 1.96 * std)
        hi = float(pred_row[h] + 1.96 * std)
        return "Y" if (real_val >= lo and real_val <= hi) else "N"

    # Build Max block
    max_real = np.max(real_w, axis=1)
    max_block = pd.DataFrame(index=idx)
    max_block[("Max", "Real")] = max_real

    for m in MODELS_6:
        max_pred = np.max(preds[m], axis=1)
        max_block[("Max", f"{m} Forecast")] = max_pred
        max_block[("Max", f"{m} Delta")] = max_pred - max_real
        max_block[("Max", f"{m} Coverage")] = [
            coverage_at_real_extreme(real_w[i], preds[m][i], vars_[m][i], "max") for i in range(days)
        ]
        

    # Build Min block
    min_real = np.min(real_w, axis=1)
    min_block = pd.DataFrame(index=idx)
    min_block[("Min", "Real")] = min_real

    for m in MODELS_6:
        min_pred = np.min(preds[m], axis=1)
        min_block[("Min", f"{m} Forecast")] = min_pred
        min_block[("Min", f"{m} Delta")] = min_pred - min_real
        min_block[("Min", f"{m} Coverage")] = [
            coverage_at_real_extreme(real_w[i], preds[m][i], vars_[m][i], "min") for i in range(days)
        ]
        

    out = pd.concat([max_block, min_block], axis=1)

    # Make column order stable / readable
    out = out.sort_index(axis=1, level=0)
    return out

def compute_dm_matrices(country: str, year: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    real_full = load_matrix_2d(path_real(country, year), n_cols=N_COLS)
    preds = []
    labels = MODELS_6[:]

    for m in MODELS_6:
        pred_path, _ = path_pred_var(country, year, m)
        pred = load_matrix_2d(pred_path, n_cols=N_COLS)
        preds.append(pred)

    n = len(preds)
    dm_stat = np.zeros((n, n), dtype=float)
    dm_pval = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            # Germany-2021 + DNN special logic (paper code)
            if (country.lower() == "germany") and (year == 2021) and (i == 2):
                r = real_full[3:, :]
                p1 = preds[i]
                p2 = preds[j][3:, :]
            elif (country.lower() == "germany") and (year == 2021) and (j == 2):
                r = real_full[3:, :]
                p1 = preds[i][3:, :]
                p2 = preds[j]
            else:
                r = real_full
                p1 = preds[i]
                p2 = preds[j]

            stat, p = diebold_mariano_test(r[:, :N_COLS], p1[:, :N_COLS], p2[:, :N_COLS], loss_function="mae", h=1)
            dm_stat[i, j] = stat
            dm_pval[i, j] = p

    return dm_stat, dm_pval, labels

def dm_combined_table(dm_stat: np.ndarray, dm_pval: np.ndarray, labels: List[str]) -> pd.DataFrame:
    n = len(labels)
    out = pd.DataFrame(index=labels, columns=labels, dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                out.iloc[i, j] = "_"
            elif i < j:
                p = float(dm_pval[i, j])
                out.iloc[i, j] = "0" if p < 1e-16 else f"{p:.2e}"
            else:
                out.iloc[i, j] = f"{float(dm_stat[i, j]):.3f}"
    return out

def tables_dm_paper():
    # tab:dm_germany_2021_2023 (three subtables)
    dm_germany = {}
    for y in [2021, 2022, 2023]:
        s, p, labs = compute_dm_matrices("germany", y)
        dm_germany[y] = (s, p, labs)

    t_g_2021 = dm_combined_table(dm_germany[2021][0], dm_germany[2021][1], dm_germany[2021][2])
    t_g_2022 = dm_combined_table(dm_germany[2022][0], dm_germany[2022][1], dm_germany[2022][2])
    t_g_2023 = dm_combined_table(dm_germany[2023][0], dm_germany[2023][1], dm_germany[2023][2])

    # tab:dm_fr_no_2023 (two subtables)
    s_fr, p_fr, labs_fr = compute_dm_matrices("france", 2023)
    s_no, p_no, labs_no = compute_dm_matrices("norway", 2023)
    t_fr_2023 = dm_combined_table(s_fr, p_fr, labs_fr)
    t_no_2023 = dm_combined_table(s_no, p_no, labs_no)

    return (t_g_2021, t_g_2022, t_g_2023, t_fr_2023, t_no_2023, dm_germany)

def friedman_posthoc_rcnp_table() -> pd.DataFrame:
    # tab:friedman_posthoc_rcnp
    cols = scenario_columns()
    df = pd.DataFrame(index=["XGB", "DNN", "LEAR", "TFT", "BLSTM"], columns=cols, dtype=float)

    for c, y, lab in SCENARIOS:
        real = load_matrix_2d(path_real(c, y), n_cols=N_COLS)
        mae_list = []

        # order: cnp, xgb, dnn, lear, tft, blstm (paper code)
        order = ["R-CNP", "XGB", "DNN", "LEAR", "TFT", "BLSTM"]
        print(lab)
        if lab == "Germany-2021":
            for m in order:
                pred_path, _ = path_pred_var(c, y, m)
                pred = load_matrix_2d(pred_path, n_cols=N_COLS)
                real_m, pred_m, _ = align_real_pred(c, y, m, real, pred, None)
                if m != "DNN":
                    mae_list.append(mae(pred_m[3:], real_m[3:], axis=1))
                else:
                    mae_list.append(mae(pred_m, real_m, axis=1))
                # make equal-length by trimming to min (defensive)
                T = min(len(v) for v in mae_list)
                mae_mat = np.column_stack([v[:T] for v in mae_list])  # shape (T,6)
            print(mae_mat.shape)
            ranks = np.argsort(np.argsort(mae_mat, axis=1), axis=1) + 1
        else:
            for m in order:
                pred_path, _ = path_pred_var(c, y, m)
                pred = load_matrix_2d(pred_path, n_cols=N_COLS)
                real_m, pred_m, _ = align_real_pred(c, y, m, real, pred, None)
                mae_list.append(mae(pred_m, real_m, axis=1))


                # make equal-length by trimming to min (defensive)
                T = min(len(v) for v in mae_list)
                mae_mat = np.column_stack([v[:T] for v in mae_list])  # shape (T,6)
            ranks = np.argsort(np.argsort(mae_mat, axis=1), axis=1) + 1

        nemenyi = sp.posthoc_nemenyi_friedman(ranks)
        nemenyi.index = ["cnp", "xgb", "dnn", "lear", "tft", "blstm"]
        nemenyi.columns = ["cnp", "xgb", "dnn", "lear", "tft", "blstm"]

        df.loc["XGB", lab] = float(nemenyi.loc["cnp", "xgb"])
        df.loc["DNN", lab] = float(nemenyi.loc["cnp", "dnn"])
        df.loc["LEAR", lab] = float(nemenyi.loc["cnp", "lear"])
        df.loc["TFT", lab] = float(nemenyi.loc["cnp", "tft"])
        df.loc["BLSTM", lab] = float(nemenyi.loc["cnp", "blstm"])

    return df

def profit_cost_means(country: str, year: int, model: str) -> Dict[str, float]:
    out = {}
    for case in ["I", "II", "III", "IV"]:
        kind = "cost" if case == "IV" else "profit"
        series = load_vector(path_trading_series(country, year, model, case, kind))
        out[case] = float(np.mean(series))
    return out

def perfect_foresight_means(country: str, year: int) -> Dict[str, float]:
    # paper uses PF line; safest: try TFT PF first then R-CNP PF
    out = {}
    for case in ["I", "II", "III", "IV"]:
        pf_series = None
        for try_model in ["TFT", "R-CNP"]:
            try:
                pf_series = load_vector(path_trading_series(country, year, try_model, case, "pf"))
                break
            except Exception:
                pf_series = None
        if pf_series is None:
            raise FileNotFoundError(f"Missing PF series for {country}-{year} case {case}")
        out[case] = float(np.mean(pf_series))
    return out

def table_profit_cost_merged_all() -> pd.DataFrame:
    # tab:profit_cost_merged_all
    cols = scenario_columns()

    # Unique case labels (fixes your crash and matches the paper structure)
    case_labels = [
        ("Case-I Mean Profit (€/Day)",  "I"),
        ("Case-II Mean Profit (€/Day)", "II"),
        ("Case-III Mean Profit (€/Day)","III"),
        ("Case-IV Mean Cost (€/Day)",   "IV"),
    ]
    models_rows = ["cnp", "dnn", "lear", "blstm", "xgb", "tft", "perfect foresight"]

    idx = pd.MultiIndex.from_product([[cl[0] for cl in case_labels], models_rows], names=["Case", "Model"])
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)

    for c, y, lab in SCENARIOS:
        pf = perfect_foresight_means(c, y)
        for case_name, case in case_labels:
            # model means
            for mrow, m in zip(["cnp","dnn","lear","blstm","xgb","tft"], ["R-CNP","DNN","LEAR","BLSTM","XGB","TFT"]):
                means = profit_cost_means(c, y, m)
                df.loc[(case_name, mrow), lab] = means[case]
            # PF
            df.loc[(case_name, "perfect foresight"), lab] = pf[case]

    return df

def table_profit_cost_gap_pf() -> pd.DataFrame:
    # tab:profit_cost_gap_pf (absolute mean gaps, exactly as paper)
    cols = scenario_columns()
    case_labels = [
        ("Case-I Profit Gap (€/Day)",   "I"),
        ("Case-II Profit Gap (€/Day)",  "II"),
        ("Case-III Profit Gap (€/Day)", "III"),
        ("Case-IV Cost Gap (€/Day)",    "IV"),
    ]
    models_rows = ["cnp", "dnn", "lear", "blstm", "xgb", "tft"]
    idx = pd.MultiIndex.from_product([[cl[0] for cl in case_labels], models_rows], names=["Case", "Model"])
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)

    for c, y, lab in SCENARIOS:
        pf = perfect_foresight_means(c, y)
        for case_name, case in case_labels:
            for mrow, m in zip(["cnp","dnn","lear","blstm","xgb","tft"], ["R-CNP","DNN","LEAR","BLSTM","XGB","TFT"]):
                means = profit_cost_means(c, y, m)
                if case != "IV":
                    # profit gap = PF - model
                    df.loc[(case_name, mrow), lab] = pf[case] - means[case]
                else:
                    # cost gap = model - PF
                    df.loc[(case_name, mrow), lab] = means[case] - pf[case]

    return df


# ======================================================================================
# TOPSIS + robustness (only what paper table shows)
# ======================================================================================

def topsis(df: pd.DataFrame, weights, benefit_criteria) -> pd.DataFrame:
    if isinstance(weights, dict):
        w = np.array([weights[c] for c in df.columns], dtype=float)
    else:
        w = np.array(weights, dtype=float)
    w = w / w.sum()

    if isinstance(benefit_criteria, dict):
        b = np.array([benefit_criteria[c] for c in df.columns], dtype=bool)
    else:
        b = np.array(benefit_criteria, dtype=bool)

    X = df.values.astype(float)
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    R = X / denom
    V = R * w

    ideal_best = np.where(b, V.max(axis=0), V.min(axis=0))
    ideal_worst = np.where(b, V.min(axis=0), V.max(axis=0))

    d_pos = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    scores = d_neg / (d_pos + d_neg + 1e-12)

    out = df.copy()
    out["TOPSIS_Score"] = scores
    out["Rank"] = (-scores).argsort().argsort() + 1
    out = out.sort_values("Rank")
    return out

def grouped_dirichlet_weights(cols, acc_cols, ops_cols, p_acc: float, rng) -> dict:
    w_acc = rng.dirichlet(np.ones(len(acc_cols))) if len(acc_cols) > 1 else np.array([1.0])
    w_ops = rng.dirichlet(np.ones(len(ops_cols))) if len(ops_cols) > 1 else np.array([1.0])
    w = {}
    for i, c in enumerate(acc_cols):
        w[c] = p_acc * w_acc[i]
    for i, c in enumerate(ops_cols):
        w[c] = (1.0 - p_acc) * w_ops[i]
    ww = np.array([w[c] for c in cols], dtype=float)
    ww = ww / ww.sum()
    return {c: ww[i] for i, c in enumerate(cols)}

def robustness_summary(decision: pd.DataFrame, scheme: str, n: int, p_acc: float, seed: int=123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = list(decision.columns)
    acc_cols = ["MAE", "RMSE", "SMAPE"]
    ops_cols = [c for c in cols if c not in acc_cols]
    models = list(decision.index)

    ranks = {m: [] for m in models}
    top1 = {m: 0 for m in models}

    benefit = {c: False for c in cols}  # ALL COST

    for _ in range(n):
        if scheme == "grouped":
            w = grouped_dirichlet_weights(cols, acc_cols, ops_cols, p_acc=p_acc, rng=rng)
        elif scheme == "full":
            ww = rng.dirichlet(np.ones(len(cols)))
            w = {c: ww[i] for i, c in enumerate(cols)}
        else:
            raise ValueError("scheme must be 'grouped' or 'full'")

        res = topsis(decision, weights=w, benefit_criteria=benefit)
        best = res.index[0]
        top1[best] += 1
        for m in models:
            ranks[m].append(int(res.loc[m, "Rank"]))

    out = pd.DataFrame(index=models, columns=["AvgR", "P(Top1)"], dtype=float)
    for m in models:
        rr = np.array(ranks[m], dtype=float)
        out.loc[m, "AvgR"] = float(rr.mean())
        out.loc[m, "P(Top1)"] = float(top1[m] / n)

    return out

def topsis_robustness_all_table() -> Dict[str, Dict[str, pd.DataFrame]]:
    # tab:topsis_robustness_all: per scenario, provide 3 robustness summaries:
    # grouped(70/30), grouped(30/70), full
    results = {}

    # precompute profit/cost gaps
    gaps_tbl = table_profit_cost_gap_pf()

    for c, y, lab in SCENARIOS:
        # decision matrix for TOPSIS models (paper order)
        rows = []
        for m in TOPSIS_MODELS:
            a, b, s = compute_error_summary(c, y, m)
            # gaps: map model name -> lower key in gaps table
            mrow = "cnp" if m == "R-CNP" else m.lower()
            # fetch gaps from gaps_tbl
            g1 = float(gaps_tbl.loc[("Case-I Profit Gap (€/Day)", mrow), lab])
            g2 = float(gaps_tbl.loc[("Case-II Profit Gap (€/Day)", mrow), lab])
            g3 = float(gaps_tbl.loc[("Case-III Profit Gap (€/Day)", mrow), lab])
            g4 = float(gaps_tbl.loc[("Case-IV Cost Gap (€/Day)", mrow), lab])

            rows.append({"Model": m, "MAE": a, "RMSE": b, "SMAPE": s,
                         "Gap_Case_I": g1, "Gap_Case_II": g2, "Gap_Case_III": g3, "Gap_Case_IV": g4})

        decision = pd.DataFrame(rows).set_index("Model")[["MAE", "RMSE", "SMAPE", "Gap_Case_I", "Gap_Case_II", "Gap_Case_III", "Gap_Case_IV"]].round(3)
        #print(decision)
        # robustness summaries (paper uses n=5000)
        r70 = robustness_summary(decision, scheme="grouped", n=5000, p_acc=0.7, seed=123)
        r30 = robustness_summary(decision, scheme="grouped", n=5000, p_acc=0.3, seed=456)
        rfull = robustness_summary(decision, scheme="full", n=5000, p_acc=0.0, seed=789)

        results[lab] = {
            "Grouped_70_30": r70,
            "Grouped_30_70": r30,
            "Full": rfull,
        }

    return results


# ======================================================================================
# MAIN – paper order
# ======================================================================================

def main():
    ensure_dirs()
    _configure_matplotlib_pdf_quality()

    # (1) tab:sensitivity_hyperparams_compact
    t1 = table_sensitivity_hyperparams_compact()
    print(t1)
    save_df_csv(t1, "tab_sensitivity_hyperparams_compact.csv")

    # (2) tab:rcnp_vs_noregimes_all
    t2 = table_rcnp_vs_noregimes_all()
    print(t2.round(3))
    save_df_csv(t2.round(6), "tab_rcnp_vs_noregimes_all.csv")

    # (3) tab:all_errors_merged_all
    t3 = table_all_errors_merged_all()
    print(t3.round(3))
    save_df_csv(t3.round(6), "tab_all_errors_merged_all.csv")

    t4 = table_picp_mpiw_merged_all()
    print(t4.round(3))
    save_df_csv(t4.round(6), "tab_picp_mpiw_merged_all.csv")

    # (4) regime plots (fig:regime_detect_germany + fig:num_regime_2021 block in paper)
    for c in ["germany", "france", "norway"]:
        plot_regime_plots_2023(c)

    # (5) models comparison Germany-2023 (fig:RNP_2023_Interval)
    plot_models_comparison_week("germany", 2023, os.path.join(IMAGES_DIR, "models_comparison_germany_2023.pdf"))

    # (6) RMSE accuracy plots (fig:RMSE_accuracy + fig:Error_France)
    plot_error_curves_rmse("germany", 2023, os.path.join(IMAGES_DIR, "error_comparison_germany_2023.pdf"))
    plot_error_curves_rmse("france",  2023, os.path.join(IMAGES_DIR, "error_comparison_france_2023.pdf"))
    plot_error_curves_rmse("norway",  2023, os.path.join(IMAGES_DIR, "error_comparison_norway_2023.pdf"))


    # (7) Failure case plot France-2023 (Failure_case figure)
    plot_models_comparison_week("france", 2023, os.path.join(IMAGES_DIR, "Failure_case_france.pdf"), week_slice=slice(1,8))

    # (8) tab:7day_extremes
    t5 = failure_case_extremes_table_france_2023()
    print(t5)
    save_df_csv(t5, "tab_7day_extremes.csv")

    # (9) DM tables + DM heatmap (tab:dm_* and fig:dm_germany)
    t_g21, t_g22, t_g23, t_fr23, t_no23, dm_germany_raw = tables_dm_paper()
    print(t_fr23)
    print(t_no23)
    save_df_csv(t_fr23, "tab_dm_fr_2023.csv")
    save_df_csv(t_no23, "tab_dm_no_2023.csv")

    print(t_g21)
    print(t_g22)
    print(t_g23)
    save_df_csv(t_g21, "tab_dm_germany_2021.csv")
    save_df_csv(t_g22, "tab_dm_germany_2022.csv")
    save_df_csv(t_g23, "tab_dm_germany_2023.csv")

    dm_pvals_by_year = {y: dm_germany_raw[y][1] for y in [2021, 2022, 2023]}
    plot_dm_panels_germany(dm_pvals_by_year, os.path.join(IMAGES_DIR, "dm_pvals_germany_2021_2023.pdf"))

    # (10) tab:friedman_posthoc_rcnp
    t8 = friedman_posthoc_rcnp_table()
    # replace exact zeros with <1e-16 for paper-style reporting
    t8_fmt = t8.copy()
    t8_fmt = t8_fmt.applymap(lambda v: ("<1e-16" if float(v) < 1e-16 else float(v)))
    print(t8_fmt)
    save_df_csv(t8, "tab_friedman_posthoc_rcnp.csv")

    # (11) tab:profit_cost_merged_all
    t9 = table_profit_cost_merged_all()
    print(t9.round(3))
    save_df_csv(t9.round(6), "tab_profit_cost_merged_all.csv")

    # (12) tab:profit_cost_gap_pf
    t10 = table_profit_cost_gap_pf()
    print(t10)
    save_df_csv(t10.round(6), "tab_profit_cost_gap_pf.csv")

    # (13) tab:topsis_robustness_all
    rob = topsis_robustness_all_table()
    # print each scenario’s three summaries (this matches the paper: separate small blocks)
    for lab in scenario_columns():
        print(rob[lab]["Grouped_70_30"].loc[TOPSIS_MODELS].round(3))
        print(rob[lab]["Grouped_30_70"].loc[TOPSIS_MODELS].round(3))
        print(rob[lab]["Full"].loc[TOPSIS_MODELS].round(3))
        save_df_csv(rob[lab]["Grouped_70_30"], f"tab_topsis_{lab}_grouped_70_30.csv")
        save_df_csv(rob[lab]["Grouped_30_70"], f"tab_topsis_{lab}_grouped_30_70.csv")
        save_df_csv(rob[lab]["Full"], f"tab_topsis_{lab}_full.csv")

    plt.show()


if __name__ == "__main__":
    warnings.simplefilter("once")
    main()



"""
# =========================
# TOPSIS (cost/benefit)
# =========================
def topsis(df: pd.DataFrame, weights, benefit_criteria) -> pd.DataFrame:
    
    if isinstance(weights, dict):
        w = np.array([weights[c] for c in df.columns], dtype=float)
    else:
        w = np.array(weights, dtype=float)

    if w.size != df.shape[1]:
        raise ValueError(f"weights length {w.size} != number of criteria {df.shape[1]}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if w.sum() == 0:
        raise ValueError("sum(weights) must be > 0")
    w = w / w.sum()

    # --- resolve benefit/cost flags ---
    if isinstance(benefit_criteria, dict):
        b = np.array([benefit_criteria[c] for c in df.columns], dtype=bool)
    else:
        b = np.array(benefit_criteria, dtype=bool)

    if b.size != df.shape[1]:
        raise ValueError(f"benefit_criteria length {b.size} != number of criteria {df.shape[1]}")

    # --- numeric matrix ---
    X = df.values.astype(float)

    # Step 1: normalize columns by Euclidean norm
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    R = X / denom

    # Step 2: apply weights
    V = R * w

    # Step 3: ideal best/worst
    ideal_best = np.where(b, V.max(axis=0), V.min(axis=0))
    ideal_worst = np.where(b, V.min(axis=0), V.max(axis=0))

    # Step 4: distances
    d_pos = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    # Step 5: score (higher is better)
    scores = d_neg / (d_pos + d_neg + 1e-12)

    out = df.copy()
    out["TOPSIS_Score"] = scores
    out["Rank"] = (-scores).argsort().argsort() + 1
    out = out.sort_values("Rank")
    return out


# =========================
# Robustness weight sampling
# =========================
def dirichlet_on_simplex(k: int, alpha: float = 1.0, rng=None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return rng.dirichlet(alpha * np.ones(k))

def grouped_dirichlet_weights(
    cols,
    acc_cols,
    ops_cols,
    p_acc: float,
    alpha_acc: float = 1.0,
    alpha_ops: float = 1.0,
    rng=None
) -> dict:
    
    #Allocate total mass p_acc to accuracy group and (1-p_acc) to operations group,
    #then distribute within each group via Dirichlet.
    
    rng = np.random.default_rng() if rng is None else rng

    w_acc = rng.dirichlet(alpha_acc * np.ones(len(acc_cols))) if len(acc_cols) > 1 else np.array([1.0])
    w_ops = rng.dirichlet(alpha_ops * np.ones(len(ops_cols))) if len(ops_cols) > 1 else np.array([1.0])

    w = {}
    for i, c in enumerate(acc_cols):
        w[c] = p_acc * w_acc[i]
    for i, c in enumerate(ops_cols):
        w[c] = (1.0 - p_acc) * w_ops[i]

    # ensure order and exact sum=1
    ww = np.array([w[c] for c in cols], dtype=float)
    ww = ww / ww.sum()
    return {c: ww[i] for i, c in enumerate(cols)}

def robustness_check(
    df: pd.DataFrame,
    benefit: dict,
    scheme: str,
    n: int = 5000,
    p_acc: float = 0.7,
    alpha_acc: float = 1.0,
    alpha_ops: float = 1.0,
    alpha_full: float = 1.0,
    seed: int = 123
):
    
    #scheme in {"grouped", "full"}.
    #- grouped: grouped Dirichlet within accuracy/operations groups with fixed p_acc
    #- full: Dirichlet over all criteria
    
    rng = np.random.default_rng(seed)

    cols = list(df.columns)
    acc_cols = ["MAE", "RMSE", "SMAPE"]
    ops_cols = [c for c in cols if c not in acc_cols]

    models = list(df.index)
    rank_counts = pd.DataFrame(0, index=models, columns=[f"Rank{i}" for i in range(1, len(models) + 1)])
    top1 = pd.Series(0, index=models, dtype=int)
    ranks_all = {m: [] for m in models}

    for _ in range(n):
        if scheme == "grouped":
            w = grouped_dirichlet_weights(cols, acc_cols, ops_cols, p_acc=p_acc,
                                          alpha_acc=alpha_acc, alpha_ops=alpha_ops, rng=rng)
        elif scheme == "full":
            ww = rng.dirichlet(alpha_full * np.ones(len(cols)))
            w = {c: ww[i] for i, c in enumerate(cols)}
        else:
            raise ValueError("scheme must be 'grouped' or 'full'")

        res = topsis(df, weights=w, benefit_criteria=benefit)
        # res is sorted by Rank
        for m in models:
            r = int(res.loc[m, "Rank"])
            rank_counts.loc[m, f"Rank{r}"] += 1
            ranks_all[m].append(r)

        best_model = res.index[0]
        top1[best_model] += 1

    top1_df = pd.DataFrame({"Top1_Freq": top1, "Top1_Prob": top1 / n}).sort_values("Top1_Prob", ascending=False)

    stats = []
    for m in models:
        rr = np.array(ranks_all[m], dtype=float)
        stats.append({
            "Model": m,
            "Avg_Rank": rr.mean(),
            "Median_Rank": float(np.median(rr)),
            "P(Top1)": float((rr == 1).mean()),
            "P(Top3)": float((rr <= 3).mean())
        })
    stats_df = pd.DataFrame(stats).sort_values(["Avg_Rank", "P(Top1)"], ascending=[True, False])

    return top1_df, rank_counts, stats_df


# ============================================================
# DATA (Errors table + NEW PF-gap table)
# ============================================================
models = ["R-CNP", "DNN", "LEAR", "XGB", "BLSTM", "TFT"]

criteria_cols = [
    "MAE", "RMSE", "SMAPE",
    "Gap_Case_I", "Gap_Case_II", "Gap_Case_III", "Gap_Case_IV"
]

# --- Error metrics (Table 3) ---
err = {
    "Germany_2021": {
        "MAE":   [14.119, 14.333, 15.271, 16.039, 18.967, 14.179],
        "RMSE":  [16.741, 17.205, 18.069, 18.616, 22.427, 16.736],
        "SMAPE": [19.548, 18.364, 18.959, 20.014, 23.673, 18.270],
    },
    "Germany_2022": {
        "MAE":   [33.185, 31.637, 39.130, 34.595, 35.111, 33.749],
        "RMSE":  [39.027, 38.171, 46.100, 40.480, 41.709, 39.609],
        "SMAPE": [21.136, 20.609, 24.325, 20.965, 21.857, 20.694],
    },
    "Germany_2023": {
        "MAE":   [15.627, 16.285, 19.241, 16.729, 30.004, 16.088],
        "RMSE":  [19.154, 19.993, 22.741, 19.789, 35.375, 19.334],
        "SMAPE": [27.889, 29.235, 30.467, 30.743, 39.994, 27.370],
    },
    "France_2023": {
        "MAE":   [20.027, 17.147, 33.999, 20.101, 23.143, 19.927],
        "RMSE":  [23.417, 20.819, 37.184, 22.858, 27.306, 23.335],
        "SMAPE": [30.163, 27.998, 40.145, 32.012, 32.150, 29.168],
    },
    "Norway_2023": {
        "MAE":   [13.130, 18.457, 24.831, 14.171, 17.508, 15.287],
        "RMSE":  [16.103, 21.390, 26.903, 16.143, 20.808, 17.952],
        "SMAPE": [26.182, 34.764, 36.570, 28.377, 30.768, 27.090],
    },
}

# --- NEW PF gaps table (Table you provided) ---
# All are COST criteria (smaller is better; 0 = PF)
gap = {
    "Germany_2021": {
        "Gap_Case_I":   [0.077, 0.073, 0.069, 0.078, 0.148, 0.071],
        "Gap_Case_II":  [0.091, 0.113, 0.090, 0.094, 0.142, 0.086],
        "Gap_Case_III": [0.090, 0.113, 0.089, 0.094, 0.141, 0.086],
        "Gap_Case_IV":  [0.034, 0.037, 0.030, 0.028, 0.045, 0.027],
    },
    "Germany_2022": {
        "Gap_Case_I":   [0.161, 0.184, 0.146, 0.166, 0.266, 0.149],
        "Gap_Case_II":  [0.174, 0.354, 0.325, 0.265, 0.262, 0.222],
        "Gap_Case_III": [0.174, 0.354, 0.325, 0.265, 0.262, 0.222],
        "Gap_Case_IV":  [0.057, 0.069, 0.073, 0.063, 0.075, 0.055],
    },
    "Germany_2023": {
        "Gap_Case_I":   [0.056, 0.076, 0.072, 0.069, 0.195, 0.052],
        "Gap_Case_II":  [0.073, 0.197, 0.118, 0.125, 0.290, 0.101],
        "Gap_Case_III": [0.073, 0.197, 0.118, 0.125, 0.290, 0.101],
        "Gap_Case_IV":  [0.025, 0.048, 0.039, 0.036, 0.112, 0.028],
    },
    "France_2023": {
        "Gap_Case_I":   [0.101, 0.111, 0.096, 0.085, 0.166, 0.108],
        "Gap_Case_II":  [0.135, 0.447, 0.189, 0.206, 0.284, 0.193],
        "Gap_Case_III": [0.135, 0.447, 0.189, 0.206, 0.284, 0.193],
        "Gap_Case_IV":  [0.048, 0.095, 0.051, 0.040, 0.091, 0.045],
    },
    "Norway_2023": {
        "Gap_Case_I":   [0.124, 0.220, 0.085, 0.076, 0.235, 0.104],
        "Gap_Case_II":  [0.172, 0.238, 0.129, 0.113, 0.252, 0.154],
        "Gap_Case_III": [0.172, 0.179, 0.129, 0.113, 0.247, 0.154],
        "Gap_Case_IV":  [0.057, 0.110, 0.036, 0.029, 0.107, 0.047],
    },
}

def build_scenario_df(scenario: str) -> pd.DataFrame:
    if scenario not in err or scenario not in gap:
        raise KeyError(f"Missing scenario '{scenario}' in err or gap dicts")

    data = {}
    for k in ["MAE", "RMSE", "SMAPE"]:
        data[k] = err[scenario][k]
    for k in ["Gap_Case_I", "Gap_Case_II", "Gap_Case_III", "Gap_Case_IV"]:
        data[k] = gap[scenario][k]

    df = pd.DataFrame(data, index=models)
    return df[criteria_cols]

# =========================
# IMPORTANT: ALL COST NOW
# =========================
benefit = {c: False for c in criteria_cols}   # ALL are "smaller is better"

# Baseline weights: equal weights (defensible default)
weights_equal = {c: 1.0 for c in criteria_cols}

scenarios = ["Germany_2021", "Germany_2022", "Germany_2023", "France_2023", "Norway_2023"]

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

for sc in scenarios:
    df_sc = build_scenario_df(sc)
    print("\n" + "=" * 100)
    print(f"SCENARIO: {sc}")
    #print("\n--- TOPSIS (equal weights), using PF-gap operations criteria (ALL COST) ---")
    #res = topsis(df_sc, weights=weights_equal, benefit_criteria=benefit)
    #print(res[criteria_cols + ["TOPSIS_Score", "Rank"]].to_string())

    # Robustness checks
    print("\n--- Robustness: GROUPED Dirichlet within groups (70/30), alpha_acc=1.0, alpha_ops=1.0, n=5000 ---")
    top1, rdist, stats = robustness_check(df_sc, benefit, scheme="grouped", n=5000, p_acc=0.7,
                                         alpha_acc=1.0, alpha_ops=1.0, seed=123)
    #print("\nTop-1 frequency (Rank 1 probability):")
    #print(top1.to_string())
    #print("\nRank distribution (counts):")
    #print(rdist.to_string())
    print("\nRank stability stats:")
    print(stats[["Model", "Avg_Rank",  "P(Top1)"]])

    print("\n--- Robustness: GROUPED Dirichlet within groups (30/70), alpha_acc=1.0, alpha_ops=1.0, n=5000 ---")
    top1, rdist, stats = robustness_check(df_sc, benefit, scheme="grouped", n=5000, p_acc=0.3,
                                         alpha_acc=1.0, alpha_ops=1.0, seed=456)
    #print("\nTop-1 frequency (Rank 1 probability):")
    #print(top1.to_string())
    #print("\nRank distribution (counts):")
    #print(rdist.to_string())
    print("\nRank stability stats:")
    print(stats[["Model", "Avg_Rank",  "P(Top1)"]])

    print("\n--- Robustness: FULL Dirichlet(alpha=1.0) on ALL criteria, n=5000 ---")
    top1, rdist, stats = robustness_check(df_sc, benefit, scheme="full", n=5000,
                                         alpha_full=1.0, seed=789)
    #print("\nTop-1 frequency (Rank 1 probability):")
    #print(top1.to_string())
    #print("\nRank distribution (counts):")
    #print(rdist.to_string())
    print("\nRank stability stats:")
    print(stats[["Model", "Avg_Rank",  "P(Top1)"]])
"""
