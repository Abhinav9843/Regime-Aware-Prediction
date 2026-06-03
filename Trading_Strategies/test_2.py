"""
Germany 2021 operational demo for CNP, LEAR, and DNN.

This script evaluates all four operational cases using the existing
`trading_strategies_I_IV.py` functions:

    Case I   : realized profit
    Case II  : realized profit
    Case III : reported profit
    Case IV  : realized cost

It also reports perfect-foresight references and a compact Case II vs Case III
diagnostic.

Edit DATA_ROOT before running.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import trading_strategies_I_IV as ts


# ---------------------------------------------------------------------
# Display settings: show all columns, not "..."
# ---------------------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

np.set_printoptions(precision=8, suppress=True)


# ---------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------

DATA_ROOT = Path(r"C:/Users/abhin/Downloads/Regime-Aware-Prediction-main")

COUNTRY = "germany"
YEAR = 2021
T = 24

# Keep False to use parameters exactly as defined in trading_strategies_I_IV.py.
# Set True only if you explicitly want to force +/- 5 kWh = +/- 0.005 MWh.
FORCE_PAPER_POWER_BOUND = False


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def show_df(df: pd.DataFrame, digits: int = 6) -> None:
    """Display a DataFrame in Jupyter if available; otherwise print all columns."""
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)

    try:
        from IPython.display import display
        display(out)
    except Exception:
        print(out.to_string(index=False))


def load_txt(path: Path, rows=None, cols=24) -> np.ndarray:
    """Load a text file and optionally slice rows/columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file:\n{path}")

    arr = np.loadtxt(path)
    arr = np.asarray(arr)

    if arr.ndim == 1:
        if arr.size < cols:
            raise ValueError(f"{path} is 1D with length {arr.size}, expected at least {cols}.")
        arr = arr[:cols].reshape(1, cols)
    else:
        arr = arr[:, :cols]

    if rows is not None:
        arr = arr[:rows, :]

    return arr


def safe_sigma(var_row: np.ndarray) -> np.ndarray:
    """Convert hourly predictive variance to hourly standard deviation."""
    return np.sqrt(np.maximum(var_row, 0.0))


def maybe_force_paper_power_bound() -> None:
    """Optionally force the imported module to use +/- 0.005 MWh power bounds."""
    if FORCE_PAPER_POWER_BOUND:
        ts.x_max_net = 0.005
        ts.x_min_net = -0.005
        ts.P_charge_max = abs(ts.x_min_net)
        ts.P_discharge_max = ts.x_max_net


def module_parameter_table() -> pd.DataFrame:
    """Show the key parameters currently used by trading_strategies_I_IV.py."""
    rows = {
        "T": getattr(ts, "T", np.nan),
        "C_max_MWh": getattr(ts, "C_max", np.nan),
        "x_min_net_MWh": getattr(ts, "x_min_net", np.nan),
        "x_max_net_MWh": getattr(ts, "x_max_net", np.nan),
        "eta_c": getattr(ts, "eta_c", np.nan),
        "eta_d": getattr(ts, "eta_d", np.nan),
        "lambda_risk": getattr(ts, "lambda_risk", np.nan),
        "mu_residual": getattr(ts, "mu_residual", np.nan),
        "gamma_renewable": getattr(ts, "gamma_renewable", np.nan),
    }
    return pd.DataFrame({"Parameter": list(rows.keys()), "Value": list(rows.values())})


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_germany_2021_common_data(data_root: Path):
    """Load Germany 2021 realized prices, load, and solar generation."""
    real_2021 = load_txt(
        data_root / "Prediction Result" / "real_data" / "real_germany_2021.txt",
        rows=365,
        cols=24,
    )

    load_2021 = load_txt(
        data_root / "Trading_Strategies" / "sim_load_data_germany_2021.txt",
        rows=365,
        cols=24,
    )

    solar_2021 = load_txt(
        data_root / "Trading_Strategies" / "sim_pv_data_germany_2021.txt",
        rows=365,
        cols=24,
    )

    return real_2021, load_2021, solar_2021


def load_model_inputs(data_root: Path):
    """
    Load only the three demo models: R-CNP, LEAR, and DNN.

    The DNN offset follows the original production script, where Germany 2021
    DNN is aligned after dropping the first three realized/load/solar days.
    """
    return {
        "R-CNP": {
            "pred": load_txt(
                data_root / "Prediction Result" / "CNP" / "Result_CNP_HYPER_Tuned"
                / "Result_germany_tuned_2021" / "pred_cnp_germany_2021.txt",
                rows=365,
                cols=24,
            ),
            "var": load_txt(
                data_root / "Prediction Result" / "CNP" / "Result_CNP_HYPER_Tuned"
                / "Result_germany_tuned_2021" / "var_cnp_germany_2021.txt",
                rows=365,
                cols=24,
            ),
            "offset": 0,
        },

        "LEAR": {
            "pred": load_txt(
                data_root / "Prediction Result" / "LEAR" / "Final_pred_LEAR_Germany_2021.txt",
                rows=365,
                cols=24,
            ),
            "var": load_txt(
                data_root / "Prediction Result" / "LEAR" / "var_lear_2021_germany.txt",
                rows=365,
                cols=24,
            ),
            "offset": 0,
        },

        "DNN": {
            "pred": load_txt(
                data_root / "Prediction Result" / "DNN" / "DNN_germany_2021.txt",
                rows=None,
                cols=24,
            ),
            "var": load_txt(
                data_root / "Prediction Result" / "DNN" / "var_dnn_2021_germany.txt",
                rows=None,
                cols=24,
            ),
            "offset": 3,
        },
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_model_germany_2021(
    model_name: str,
    pred: np.ndarray,
    var: np.ndarray,
    offset: int,
    real_2021: np.ndarray,
    load_2021: np.ndarray,
    solar_2021: np.ndarray,
) -> dict:
    """
    Evaluate one forecasting model for Germany 2021.

    Cases I-III return realized/reported profit.
    Case IV returns realized cost.
    """
    if offset == 0:
        real = real_2021
        load = load_2021
        solar = solar_2021
    else:
        real = real_2021[offset:, :]
        load = load_2021[offset:, :]
        solar = solar_2021[offset:, :]

    n_days = min(pred.shape[0], var.shape[0], real.shape[0], load.shape[0], solar.shape[0])

    pred = pred[:n_days, :24]
    var = var[:n_days, :24]
    real = real[:n_days, :24]
    load = load[:n_days, :24]
    solar = solar[:n_days, :24]

    case_i = np.full(n_days, np.nan)
    case_ii = np.full(n_days, np.nan)
    case_iii = np.full(n_days, np.nan)
    case_iv = np.full(n_days, np.nan)

    pf_case_i = np.full(n_days, np.nan)
    pf_case_ii = np.full(n_days, np.nan)
    pf_case_iii = np.full(n_days, np.nan)
    pf_case_iv = np.full(n_days, np.nan)

    max_xplus_diff_ii_iii = np.full(n_days, np.nan)
    max_xminus_diff_ii_iii = np.full(n_days, np.nan)
    support_value_iii = np.full(n_days, np.nan)

    fixed_load = ts.fixed_load

    for d in range(n_days):
        predicted_prices = pred[d]
        realized_prices = real[d]
        sigma = safe_sigma(var[d])

        realized_load = load[d]
        realized_solar = solar[d]

        # Case I
        case_i[d], _, _, _ = ts.solve_case_i(
            predicted_prices,
            realized_prices,
            is_perfect_foresight=False,
        )

        pf_case_i[d], _, _, _ = ts.solve_case_i(
            realized_prices,
            realized_prices,
            is_perfect_foresight=True,
        )

        # Case II
        case_ii[d], xp_ii, xm_ii, _ = ts.solve_case_ii(
            predicted_prices,
            realized_prices,
            sigma,
            is_perfect_foresight=False,
        )

        pf_case_ii[d], _, _, _ = ts.solve_case_ii(
            realized_prices,
            realized_prices,
            sigma,
            is_perfect_foresight=True,
        )

        # Case III
        case_iii[d], xp_iii, xm_iii, _ = ts.solve_case_iii(
            predicted_prices,
            realized_prices,
            realized_load,
            realized_solar,
            sigma,
            realized_load,
            realized_solar,
            is_perfect_foresight=False,
        )

        pf_case_iii[d], _, _, _ = ts.solve_case_iii(
            realized_prices,
            realized_prices,
            realized_load,
            realized_solar,
            sigma,
            realized_load,
            realized_solar,
            is_perfect_foresight=True,
        )

        # Case IV
        case_iv[d], _, _, _, _ = ts.solve_case_iv(
            predicted_prices,
            realized_prices,
            sigma,
            fixed_load,
            is_perfect_foresight=False,
        )

        pf_case_iv[d], _, _, _, _ = ts.solve_case_iv(
            realized_prices,
            realized_prices,
            sigma,
            fixed_load,
            is_perfect_foresight=True,
        )

        # Compact Case II vs Case III diagnostic
        if xp_ii is not None and xp_iii is not None:
            q = -ts.mu_residual * realized_load + ts.gamma_renewable * realized_solar
            max_xplus_diff_ii_iii[d] = np.max(np.abs(xp_ii - xp_iii))
            max_xminus_diff_ii_iii[d] = np.max(np.abs(xm_ii - xm_iii))
            support_value_iii[d] = np.sum(q * xp_iii)

    return {
        "Model": model_name,
        "Days evaluated": n_days,

        "Case I mean profit": np.nanmean(case_i),
        "Case II mean profit": np.nanmean(case_ii),
        "Case III mean profit": np.nanmean(case_iii),
        "Case IV mean cost": np.nanmean(case_iv),

        "PF Case I mean profit": np.nanmean(pf_case_i),
        "PF Case II mean profit": np.nanmean(pf_case_ii),
        "PF Case III mean profit": np.nanmean(pf_case_iii),
        "PF Case IV mean cost": np.nanmean(pf_case_iv),

        "Mean |Case II - Case III profit|": np.nanmean(np.abs(case_ii - case_iii)),
        "Mean max |x_plus II - III|": np.nanmean(max_xplus_diff_ii_iii),
        "Mean max |x_minus II - III|": np.nanmean(max_xminus_diff_ii_iii),
        "Mean Case III support value": np.nanmean(support_value_iii),
    }


def main() -> None:
    maybe_force_paper_power_bound()

    print("Germany 2021 operational demo: R-CNP, LEAR, DNN")
    print("DATA_ROOT:", DATA_ROOT)
    print("FORCE_PAPER_POWER_BOUND:", FORCE_PAPER_POWER_BOUND)
    print()

    print("Module parameters used:")
    show_df(module_parameter_table(), digits=8)
    print()

    real_2021, load_2021, solar_2021 = load_germany_2021_common_data(DATA_ROOT)
    models = load_model_inputs(DATA_ROOT)

    print("Loaded common data shapes:")
    print("real_2021:", real_2021.shape)
    print("load_2021:", load_2021.shape)
    print("solar_2021:", solar_2021.shape)
    print()

    rows = []
    for model_name, obj in models.items():
        print(f"Evaluating {model_name}...")
        rows.append(
            evaluate_model_germany_2021(
                model_name=model_name,
                pred=obj["pred"],
                var=obj["var"],
                offset=obj["offset"],
                real_2021=real_2021,
                load_2021=load_2021,
                solar_2021=solar_2021,
            )
        )

    summary = pd.DataFrame(rows)

    # One full table with all columns visible
    print("\nGermany 2021: full operational summary")
    show_df(summary, digits=8)

    # Optional compact view
    compact = summary[
        [
            "Model",
            "Days evaluated",
            "Case I mean profit",
            "Case II mean profit",
            "Case III mean profit",
            "Case IV mean cost",
        ]
    ].copy()

    print("\nGermany 2021: compact Cases I-IV table")
    show_df(compact, digits=8)


if __name__ == "__main__":
    main()
