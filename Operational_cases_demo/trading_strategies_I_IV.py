import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set random seed for reproducibility
np.random.seed(42)

T = 24
C_max = 0.01        # maximum capacity
x_max_net = 0.003  # Max net power flow to the grid (MW)
x_min_net = -0.003 # Min net power flow to the grid (MW)
eta_c = 0.95        # charging efficiency
eta_d = 0.95        # discharging efficiency
s_init = 0       # initial SoC
s_final = 0      # desired final SoC (oracle only)

# Risk aversion and economic weights
lambda_risk = 0.5
mu_residual = 1.11
gamma_renewable = 1.67
L_total = 0.01
fixed_load = (L_total / T) + 0.0001 * np.sin(np.linspace(0, 2 * np.pi, T)) + np.random.normal(0, 0.00005, T)
fixed_load[fixed_load < 0.0001] = 0.0001 # Ensure positive load
P_charge_max = abs(x_min_net) # Maximum power capacity for charging (kW) - using magnitude of x_min_net
P_discharge_max = x_max_net    # Maximum power capacity for discharging (kW) - using x_max_net


# Time horizon (24 hours)
def remove_nan(data_set):
    for i in range(data_set.shape[0]):
        #print(i)
        prc_row = data_set[i,:]
        if np.count_nonzero(np.isnan(prc_row)) == 24 and i == 0:
            prc_row = (data_set[i+1,:] + data_set[i+2,:])/2
        elif np.count_nonzero(np.isnan(prc_row)) == 24 and i > 0 and i < (data_set.shape[0] - 1):
            prc_row = (data_set[i-1,:] + data_set[i+1,:])/2
        elif np.count_nonzero(np.isnan(prc_row)) == 24 and i == (data_set.shape[0] - 1):
            prc_row = (data_set[i-1,:] + data_set[i-2,:])/2
        elif np.count_nonzero(np.isnan(prc_row)) == 24 and i == (data_set.shape[0] - 2):
            prc_row = (data_set[i+1,:] + data_set[i+2,:])/2
        elif np.count_nonzero(np.isnan(prc_row)) > 0 and np.count_nonzero(np.isnan(prc_row)) == 23:
            #print("Detected")
            prc_row = (data_set[i-1,:] + data_set[i+1,:])/2
        elif np.count_nonzero(np.isnan(prc_row)) > 0 and np.count_nonzero(np.isnan(prc_row)) < 24 and np.argwhere(np.isnan(prc_row))[0][0] != 23:
            prc_temp = np.argwhere(np.isnan(prc_row))[0][0]
            prc_row[prc_temp] = (prc_row[prc_temp-1] + prc_row[prc_temp+1])/2
        elif np.count_nonzero(np.isnan(prc_row)) > 0 and np.count_nonzero(np.isnan(prc_row)) < 24 and np.argwhere(np.isnan(prc_row))[0][0] != 23 and np.argwhere(np.isnan(prc_row))[0][0] == 0:
            prc_temp = np.argwhere(np.isnan(prc_row))[0][0]
            prc_row[prc_temp] = (prc_row[prc_temp+1] + prc_row[prc_temp+2])/2
        elif np.count_nonzero(np.isnan(prc_row)) > 0 and np.count_nonzero(np.isnan(prc_row)) < 24 and np.argwhere(np.isnan(prc_row))[0][0] == 23:
            prc_temp = np.argwhere(np.isnan(prc_row))[0][0]
            prc_row[prc_temp] = (prc_row[prc_temp-1] + prc_row[prc_temp-2])/2
        data_set[i,:] = prc_row
    return data_set

def stand_scaler(data_set):
    for i in range(data_set.shape[0]):
        rows = data_set[i]
        scaled_rows = (rows)/(np.sum(rows))
        data_set[i] = scaled_rows
    return data_set


def solve_case_i(prices, realized_prices, is_perfect_foresight=False):
    """
    Solves the optimization problem for Case I: Trading Strategy with Predicted Price Only.
    Maximizes profit from price arbitrage.

    Args:
        prices (np.array): Array of electricity prices (predicted or realized).
        is_perfect_foresight (bool): True if solving for perfect foresight, False otherwise.

    Returns:
        tuple: (profit, x_plus_optimal, x_minus_optimal, s_optimal) if successful, else (None, None, None, None)
    """
    # Define optimization variables
    x_plus = cp.Variable(T, nonneg=True)  # Amount of electricity charged (kWh)
    x_minus = cp.Variable(T, nonneg=True) # Amount of electricity discharged (kWh)
    s = cp.Variable(T + 1)                 # State of charge (kWh), s[0] is s_init, s[T] is s_final_hour

    # Objective function: Maximize profit
    # Sum (price * discharged_energy * eta_d - price * charged_energy / eta_c)
    objective = cp.Maximize(cp.sum(prices @ (x_minus * eta_d - x_plus / eta_c)))

    # Constraints
    constraints = [
        s[0] == s_init,
        # Battery dynamics
        s[1:] == s[:-1] + eta_c * x_plus - (1 / eta_d) * x_minus,
        # State of charge limits
        s >= 0,
        s <= C_max,
        # Final state of charge
        s[T] >= s_final,
        # Power limits (charging/discharging)
        x_plus <= P_charge_max,
        x_minus <= P_discharge_max,
        # Net power flow limits (to grid)
        x_min_net <= x_minus - x_plus,
        x_minus - x_plus <= x_max_net
    ]

    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            # Calculate the profit using realized prices for performance evaluation
            actual_profit_objective = cp.sum(realized_prices @ (x_minus.value * eta_d - x_plus.value / eta_c)).value
            return actual_profit_objective, x_plus.value, x_minus.value, s.value
        else:
            print(f"Case I ({'PF' if is_perfect_foresight else 'Model'}): Problem status: {problem.status}")
            return None, None, None, None
    except Exception as e:
        print(f"Case I ({'PF' if is_perfect_foresight else 'Model'}): Error solving problem: {e}")
        return None, None, None, None

def solve_case_ii(prices, realized_prices, uncertainty, is_perfect_foresight=False):
    """
    Solves the optimization problem for Case II: Trading Strategy with Predicted Price and Uncertainty.
    Maximizes profit with a risk aversion penalty.

    Args:
        prices (np.array): Array of electricity prices (predicted or realized).
        uncertainty (np.array): Array of uncertainty estimates (standard deviations).
        is_perfect_foresight (bool): True if solving for perfect foresight, False otherwise.

    Returns:
        tuple: (profit, x_plus_optimal, x_minus_optimal, s_optimal) if successful, else (None, None, None, None)
    """
    # Define optimization variables
    x_plus = cp.Variable(T, nonneg=True)  # Amount of electricity charged (kWh)
    x_minus = cp.Variable(T, nonneg=True) # Amount of electricity discharged (kWh)
    s = cp.Variable(T + 1)                 # State of charge (kWh)

    # Objective function: Maximize profit - uncertainty penalty
    # The perfect foresight case should NOT include the uncertainty penalty.
    if is_perfect_foresight:
        objective = cp.Maximize(cp.sum(prices @ (x_minus * eta_d - x_plus / eta_c)))
    else:
        objective = cp.Maximize(
            cp.sum(prices @ (x_minus * eta_d - x_plus / eta_c) - \
                   lambda_risk * uncertainty @ (x_minus + x_plus))
        )

    # Constraints (identical to Case I)
    constraints = [
        s[0] == s_init,
        s[1:] == s[:-1] + eta_c * x_plus - (1 / eta_d) * x_minus,
        s >= 0,
        s <= C_max,
        s[T] >= s_final,
        x_plus <= P_charge_max,
        x_minus <= P_discharge_max,
        x_min_net <= x_minus - x_plus,
        x_minus - x_plus <= x_max_net
    ]

    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            # For performance evaluation, calculate actual profit using optimal decisions
            # and REALIZED prices, without the uncertainty penalty.
            actual_profit_objective = cp.sum(realized_prices @ (x_minus.value * eta_d - x_plus.value / eta_c)).value
            return actual_profit_objective, x_plus.value, x_minus.value, s.value
        else:
            print(f"Case II ({'PF' if is_perfect_foresight else 'Model'}): Problem status: {problem.status}")
            return None, None, None, None
    except Exception as e:
        print(f"Case II ({'PF' if is_perfect_foresight else 'Model'}): Error solving problem: {e}")
        return None, None, None, None

def solve_case_iii(prices, realized_prices, realized_load, realized_renewable_availability, uncertainty, load, renewable_availability, is_perfect_foresight=False):
    """
    Solves the optimization problem for Case III: Trading Strategy with Price, Uncertainty, Solar, and Residual Load.
    Maximizes profit with risk aversion and grid support terms.

    Args:
        prices (np.array): Array of electricity prices (predicted or realized).
        uncertainty (np.array): Array of uncertainty estimates (standard deviations).
        load (np.array): Array of residual load values (predicted or realized).
        renewable_availability (np.array): Array of renewable availability values (predicted or realized).
        is_perfect_foresight (bool): True if solving for perfect foresight, False otherwise.

    Returns:
        tuple: (profit, x_plus_optimal, x_minus_optimal, s_optimal) if successful, else (None, None, None, None)
    """
    # Define optimization variables
    x_plus = cp.Variable(T, nonneg=True)  # Amount of electricity charged (kWh)
    x_minus = cp.Variable(T, nonneg=True) # Amount of electricity discharged (kWh)
    s = cp.Variable(T + 1)                 # State of charge (kWh)

    # Objective function
    # Perfect foresight uses realized values for all exogenous terms and no uncertainty penalty
    if is_perfect_foresight:
        objective = cp.Maximize(
            cp.sum(prices @ (x_minus * eta_d - x_plus / eta_c) - \
                   mu_residual * load @ x_plus + \
                   gamma_renewable * renewable_availability @ x_plus)
        )
    else:
        objective = cp.Maximize(
            cp.sum(prices @ (x_minus * eta_d - x_plus / eta_c) - \
                   lambda_risk * uncertainty @ (x_plus + x_minus) - \
                   mu_residual * load @ x_plus + \
                   gamma_renewable * renewable_availability @ x_plus)
        )

    # Constraints (identical to previous cases)
    constraints = [
        s[0] == s_init,
        s[1:] == s[:-1] + eta_c * x_plus - (1 / eta_d) * x_minus,
        s >= 0,
        s <= C_max,
        s[T] >= s_final,
        x_plus <= P_charge_max,
        x_minus <= P_discharge_max,
        x_min_net <= x_minus - x_plus,
        x_minus - x_plus <= x_max_net
    ]

    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            # For performance evaluation, calculate actual profit using optimal decisions
            # and REALIZED prices, residual load, and renewable availability,
            # omitting uncertainty penalty.
            actual_profit_objective = cp.sum(
                realized_prices @ (x_minus.value * eta_d - x_plus.value / eta_c) - \
                mu_residual * realized_load @ x_plus.value + \
                gamma_renewable * realized_renewable_availability @ x_plus.value
            ).value
            return actual_profit_objective, x_plus.value, x_minus.value, s.value
        else:
            print(f"Case III ({'PF' if is_perfect_foresight else 'Model'}): Problem status: {problem.status}")
            return None, None, None, None
    except Exception as e:
        print(f"Case III ({'PF' if is_perfect_foresight else 'Model'}): Error solving problem: {e}")
        return None, None, None, None

def solve_case_iv(prices, realized_prices, uncertainty, fixed_load, is_perfect_foresight=False):
    """
    Solves the optimization problem for Case IV: Minimize Cost of Serving Fixed Load.

    Args:
        prices (np.array): Array of electricity prices (predicted or realized).
        uncertainty (np.array): Array of uncertainty estimates (standard deviations).
        fixed_load (np.array): Array of fixed hourly demand (kWh).
        is_perfect_foresight (bool): True if solving for perfect foresight, False otherwise.

    Returns:
        tuple: (cost, x_grid_buy_optimal, x_batt_to_load_optimal, x_batt_from_grid_optimal, s_optimal) if successful, else (None, None, None, None, None)
    """
    # Define optimization variables
    x_grid_buy = cp.Variable(T, nonneg=True)       # Electricity bought directly from grid for load (kWh)
    x_batt_to_load = cp.Variable(T, nonneg=True)   # Electricity discharged from battery to load (kWh)
    x_batt_from_grid = cp.Variable(T, nonneg=True) # Electricity charged into battery from grid (kWh)
    s = cp.Variable(T + 1)                         # State of charge (kWh)

    # Objective function: Minimize cost
    # Perfect foresight uses realized prices and no uncertainty penalty
    if is_perfect_foresight:
        objective = cp.Minimize(
            cp.sum(prices @ x_grid_buy + prices @ x_batt_from_grid)
        )
    else:
        objective = cp.Minimize(
            cp.sum(prices @ x_grid_buy + prices @ x_batt_from_grid + \
                   lambda_risk * uncertainty @ (x_grid_buy + x_batt_from_grid + x_batt_to_load))
        )

    # Constraints
    constraints = [
        s[0] == s_init,
        # Battery dynamics
        s[1:] == s[:-1] + eta_c * x_batt_from_grid - (1 / eta_d) * x_batt_to_load,
        # State of charge limits
        s >= 0,
        s <= C_max,
        # Final state of charge
        s[T] >= s_final,
        # Power limits (charging/discharging for battery operations)
        x_batt_from_grid <= P_charge_max,
        x_batt_to_load <= P_discharge_max,
        # Load satisfaction constraint
        # Energy from grid directly + energy from battery (after discharge efficiency) = Load
        x_grid_buy + x_batt_to_load * eta_d == fixed_load
    ]

    # Create and solve the problem
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
        if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            # For performance evaluation, calculate actual cost using optimal decisions
            # and REALIZED prices, omitting uncertainty penalty.
            actual_cost_objective = cp.sum(
                realized_prices @ x_grid_buy.value + realized_prices @ x_batt_from_grid.value
            ).value
            return actual_cost_objective, x_grid_buy.value, x_batt_to_load.value, x_batt_from_grid.value, s.value
        else:
            print(f"Case IV ({'PF' if is_perfect_foresight else 'Model'}): Problem status: {problem.status}")
            return None, None, None, None, None
    except Exception as e:
        print(f"Case IV ({'PF' if is_perfect_foresight else 'Model'}): Error solving problem: {e}")
        return None, None, None, None, None
