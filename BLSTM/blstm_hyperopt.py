from pytictoc import TicToc
t=TicToc()
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import matplotlib.colors as mcolors    
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import math

from pathlib import Path
import argparse
import os
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

warnings.simplefilter('ignore', InterpolationWarning)
warnings.simplefilter("ignore")
Xscaler_regime = StandardScaler()
Yscaler_regime = StandardScaler()
Xscaler_price  = StandardScaler()
Yscaler_price  = StandardScaler()

# Example of synthetic data generation
#np.random.seed(42)

def load_data(file_path):
    return pd.read_excel(file_path, engine='openpyxl')

def get_date_input(prompt):
    while True:
        try:
            date_str = input(prompt + " (dd.mm.yyyy): ")
            date_obj = datetime.strptime(date_str, "%d.%m.%Y")
            return date_obj
        except ValueError:
            print("Invalid date format. Please enter the date in dd.mm.yyyy format.")
            

def get_days_of_week(date_str,num_days_past):
    dd = num_days_past+1
    date_obj = datetime.strptime(date_str, "%d.%m.%Y")

    # Initialize a list to store the arrays for the past 30 days
    week_arrays = np.zeros([dd,7])
    current_date = date_obj - timedelta(days=0)
    weekday = current_date.weekday()  # Get the weekday number (0=Monday, ..., 6=Sunday)

    # Create a numpy array of 0's and 1's
    week_array = np.zeros(7, dtype=int)
    week_array[weekday] = 1
    week_arrays[-1] = week_array
    #print(current_date,week_array)

    # Generate the numpy arrays for the current date and the previous 30 days
    for i in range(1,dd):
        current_date = date_obj - timedelta(days=i)  # Calculate the date 'i' days before the input date

        weekday = current_date.weekday()  # Get the weekday number (0=Monday, ..., 6=Sunday)

        # Create a numpy array of 0's and 1's
        week_array = np.zeros(7, dtype=int)
        week_array[weekday] = 1
        week_arrays[dd-i-1,] = week_array
        #print(current_date,week_array)

    return week_arrays


def to_float(data_set):
    for i in range(data_set.shape[0]):
        vec = data_set[i,:]
        vec_float = np.array([float(x.replace(',', '').strip()) if isinstance(x, str) else float(x) for x in vec])
        data_set[i,:] = vec_float
    return(data_set)

def create_date_array(start_date, end_date):
    dates_array = []
    current_date = start_date
    while current_date <= end_date:
        dates_array.append(current_date.strftime("%d.%m.%Y"))
        current_date += timedelta(days=1)
    return dates_array

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

def center_data(data_set):
    men = np.zeros([data_set.shape[1],1])
    sd = np.zeros([data_set.shape[1],1])
    for j in range(data_set.shape[1]):
        men[j,:] = np.mean(data_set[:,j])
        sd[j,:] = np.std(data_set[:,j])
        data_set[:,j] = (data_set[:,j]-np.mean(data_set[:,j]))/np.std(data_set[:,j])
    return data_set, men, sd

country = str(input("Enter Country: "))
start_date = get_date_input("Enter the start date")
end_date = get_date_input("Enter the end date")
dates_array = create_date_array(start_date, end_date)
num_days_past = int(input("Enter Number of Days for Training: "))


t.tic()

def Train_set_Regime(price_vector, X_exo, Y_exo,
                     X_exo_pred, Y_exo_pred, date):
    
    N = price_vector.shape[0]
    XTrain_all = np.zeros((N, 11))
    YTrain_all = np.zeros((N, 1))
    days = get_days_of_week(date, num_days_past)
    time_points = np.linspace(0, num_days_past, N)
 
    for i in range(7, N):
        XTrain_all[i] = np.concatenate([
            time_points[i].reshape(1,),
            np.array([np.mean(price_vector[i - 1])]),
            np.array([np.mean(price_vector[i - 2])]),
            np.array([np.mean(price_vector[i - 3])]),
            np.array([np.mean(price_vector[i - 7])]),
            np.array([np.mean(X_exo[i])]),
            np.array([np.mean(X_exo[i - 1])]),
            np.array([np.mean(X_exo[i - 7])]),
            np.array([np.mean(Y_exo[i])]),
            np.array([np.mean(Y_exo[i - 1])]),
            np.array([np.mean(Y_exo[i - 7])]),
        ])
        YTrain_all[i] = np.mean(price_vector[i])
 
    # ── Fix 2: fit scaler on training rows only ──
    train_X = XTrain_all[7:N]
    train_Y = YTrain_all[7:N]
    XTrain_scaled = Xscaler_regime.fit_transform(train_X)
    YTrain_scaled = Yscaler_regime.fit_transform(train_Y)
 
    # ── Fix 1: build XTest from last historical + day-ahead exo ──
    tp_pred = num_days_past
    XTest_raw = np.array([[
        tp_pred,
        np.mean(price_vector[-1]),
        np.mean(price_vector[-2]),
        np.mean(price_vector[-3]),
        np.mean(price_vector[-7]),
        np.mean(X_exo_pred),
        np.mean(X_exo[-1]),
        np.mean(X_exo[-7]),
        np.mean(Y_exo_pred),
        np.mean(Y_exo[-1]),
        np.mean(Y_exo[-7]),
    ]])
    XTest = Xscaler_regime.transform(XTest_raw).flatten()
 
    return XTrain_scaled, YTrain_scaled, XTest

def create_XTrain_YTrain(price_vector, X_exo, Y_exo,
                         X_exo_pred, Y_exo_pred, date):
    
    N = price_vector.shape[0]
    XTrain_all = np.zeros((N, 241))
    YTrain_all = np.zeros((N, 48))
    days = get_days_of_week(date, num_days_past)
    time_points = np.linspace(0, num_days_past, N)
 
    # last usable i is N-2 (need both i and i+1 to be historical)
    for i in range(7, N - 1):
        XTrain_all[i] = np.concatenate([
            time_points[i].reshape(1,),
            price_vector[i - 1], price_vector[i - 2],
            price_vector[i - 3], price_vector[i - 7],
            X_exo[i], X_exo[i - 1], X_exo[i - 7],
            Y_exo[i], Y_exo[i - 1], Y_exo[i - 7],
        ])
        YTrain_all[i] = np.concatenate([price_vector[i], price_vector[i + 1]])
 
    
    train_slice = slice(7, N - 1)
    XTrain_scaled = Xscaler_price.fit_transform(XTrain_all[train_slice])
    YTrain_scaled = Yscaler_price.fit_transform(YTrain_all[train_slice])
 
    # ── Fix 1: build XTest from last historical + day-ahead exo ──
    tp_pred = num_days_past
    XTest_raw = np.concatenate([
        np.array([tp_pred]),
        price_vector[-1], price_vector[-2],
        price_vector[-3], price_vector[-7],
        X_exo_pred, X_exo[-1], X_exo[-7],
        Y_exo_pred, Y_exo[-1], Y_exo[-7],
    ]).reshape(1, 241)
    XTest_scaled = Xscaler_price.transform(XTest_raw)
 
    # append day-of-week encoding
    XTest  = np.concatenate([XTest_scaled,  days[-1].reshape(1, 7)], axis=1)
    XTrain = np.concatenate([XTrain_scaled, days[7:N - 1]],          axis=1)
    YTrain = YTrain_scaled
 
    XTrain_raw = np.concatenate([XTrain_all[train_slice], days[7:N - 1]], axis=1)
 
    return XTrain, XTrain_raw, YTrain, XTest
    
def create_data_set(data_from_excel,dates_array, num_days_past = num_days_past):
    pred_data_all = []
    real_data_all = []
    pred_arima_all = []
    pred_all = []
    data_all = []
    observed_days_all = []
    mn_all = []
    sd_all = []
    data_no_nan_all = []
    if type(data_from_excel.iloc[:,0].values[0]) == np.datetime64:
        date_col = data_from_excel.iloc[:, 0]
        date_column = date_col.dt.strftime('%d.%m.%Y').values.astype(str)
        date_diff_format = date_col.dt.strftime('%d.%m').values.astype(str)
        data = data_from_excel.iloc[:, 1:]
        year_length = 4
        for kl in range(len(dates_array)):
            date_pred = datetime.strptime(dates_array[kl], "%d.%m.%Y")
            frm_year = date_pred.year - year_length
            to_date = (date_pred - timedelta(days=1)).strftime("%d.%m.%Y")
            frm_date = ((date_pred - timedelta(days=1))-timedelta(days=1460)).strftime("%d.%m.%Y")
            start_idx = np.where(date_column == frm_date)
            end_idx = np.where(date_column == to_date)
            sub_set_data = data_from_excel.iloc[start_idx[0][0]:end_idx[0][0] + 1, :]
            sub_date_diff_format = date_diff_format[start_idx[0][0]:end_idx[0][0] + 1]
            temp1 = sub_set_data
            sub_set_data_2 = temp1.drop('Delivery day', axis=1)
            sub_set_data_2.insert(0,'date',sub_date_diff_format,True)
            dd1 = date_pred.strftime('%d.%m')
            data = sub_set_data_2.iloc[:, 1:].values
            data_no_nan = remove_nan(data.copy())
            #print(data_no_nan.shape, to_date, frm_date)
            centered_data,mn,sd = center_data(data_no_nan.copy())
            real_data_location = np.where(date_column == dates_array[kl])[0][0]
            real_data_location_all = np.arange(real_data_location,real_data_location + 1)
            real_data = np.zeros([data.shape[1], 1])
            for i in range(1):
                real_data[:,i] = data_from_excel.iloc[real_data_location_all[i],:][1:25]
            #real_data = remove_nan(real_data.T)
            real_data_all.append(real_data)
            mn_all.append(mn)
            sd_all.append(sd)
            data_all.append(centered_data)
            data_no_nan_all.append(data_no_nan[-num_days_past:].reshape(-1,1))
            
    if type(data_from_excel.iloc[:,0].values[0]) == str:
        date_column = data_from_excel.iloc[:,0].values
        split_function = np.vectorize(lambda x: x.split('.')[0] + '.' + x.split('.')[1])
        date_diff_format = split_function(date_column)
        year_length = 4
        for kl in range(len(dates_array)):
            date_pred = datetime.strptime(dates_array[kl], "%d.%m.%Y")
            frm_year = date_pred.year - year_length
            to_date = (date_pred - timedelta(days=1)).strftime("%d.%m.%Y")
            frm_date = ((date_pred - timedelta(days=1))-timedelta(days=1460)).strftime("%d.%m.%Y")
            data_frm = np.where(data_from_excel.iloc[:,0] == frm_date)
            data_to = np.where(data_from_excel.iloc[:,0] == to_date)
            
            data = data_from_excel.iloc[data_frm[0][0]:data_to[0][0]+1,1:].values
            to_float_data = to_float(data.copy())
            data_no_nan = remove_nan(to_float_data.astype(float))
            centered_data, men, sd = center_data(data_no_nan.copy())
            centered_data,mn,sd = center_data(data_no_nan.copy())
            real_data_location = np.where(date_column == dates_array[kl])[0][0]
            real_data_location_all = np.arange(real_data_location,real_data_location + 1)
            real_data = np.zeros([data.shape[1], 1])
            for i in range(1):
                real_data[:,i] = (to_float(data_from_excel.iloc[real_data_location_all[i],:][1:25].values.reshape(24,1))).reshape(24,)
            #real_data = remove_nan(real_data.T)
            real_data_all.append(real_data)
            data_all.append(centered_data)
            mn_all.append(mn)
            sd_all.append(sd)
            data_no_nan_all.append(data_no_nan[-num_days_past:].reshape(-1,1))
    return data_all, mn_all , sd_all, real_data_all, data_no_nan_all
 #################################################################################
 ###############################################################################


#### Loading data from the computer drive
#file_path_price = input('File Path For Price Data:\t')
#file_path_fore_gen = input('File Path For Forecasted Generation Data:\t\t')
#file_path_fore_res_load = input('File Path For Forecasted Residual Load Data:\t')


"""
price_data_from_excel = load_data('PowerAll_updated.xlsx')
fore_gen_data = load_data('Forecasted_generation_Day-Ahead_2015_2024.xlsx')
forecasted_residual_load = load_data('Forecasted_residual_load.xlsx')
forecasted_total_renewable =  pd.DataFrame(fore_gen_data.iloc[9:,3].values.reshape(3378,24))
forecasted_total_renewable.insert(0,'date',forecasted_residual_load.iloc[:,0].values.astype(str).reshape(-1,1),True)
"""

data_dir = BASE_DIR.as_posix() + '/Datasets/' + country + '_Dataset/'

prc = country + '_aday_ahead_price.xlsx'
genr = country + '_aday_ahead_Total_Generation_Forecast.xlsx'

if country == 'germany':
    date_len = 365
    lod = country + '_aday_ahead_Total_Residual_Load_Forecast.xlsx'
    forecasted_residual_load = load_data(data_dir + lod)
    fore_gen_data = load_data(data_dir+genr)
    forecasted_total_renewable =  pd.DataFrame(fore_gen_data.iloc[9:,3].values.reshape(3378,24))
    forecasted_total_renewable.insert(0,'date',forecasted_residual_load.iloc[:,0].values.astype(str).reshape(-1,1),True)
else:
    date_len = 364
    lod = country + '_aday_ahead_Total_Load_Forecast.xlsx'
    forecasted_residual_load = load_data(data_dir + lod)
    forecasted_total_renewable = load_data(data_dir + genr)

price_data_from_excel = load_data(data_dir+prc)

data_all_price, mn_all, sd_all,real_data_price_all,data_no_nan_price = create_data_set(price_data_from_excel,dates_array, num_days_past)
data_all_fore_load,_,_,real_data_fore_load_all,data_no_nan_fore_load = create_data_set(forecasted_residual_load,dates_array, num_days_past)
data_all_fore_total_renewable,_,_,real_data_fore_total_renewable_all,data_no_nan_fore_total_renewable = create_data_set(forecasted_total_renewable,dates_array, num_days_past)

test_length = 48
data_length = data_no_nan_price[0].shape[0]+test_length
train_length = data_length - test_length
pred_gpr = np.zeros([len(dates_array),48])
var_gpr = np.zeros([len(dates_array),48])
real = np.zeros([len(dates_array),48])


pred_price = np.zeros([len(dates_array),48])
#pred_rat = np.zeros([len(dates_array),48])
var_price = np.zeros([len(dates_array),48])
#var_rat = np.zeros([len(dates_array),48])
posterior_probs_all = []
regimes_all = []
pred_price_rescaled_all = []
iters = 800
regime_length = np.zeros([len(dates_array),iters])
assigned_regimes = np.zeros([len(dates_array),int(train_length/24 -7)])
pi_pred = []

real_data_fore_load_all_arr = np.array(real_data_fore_load_all).reshape(len(dates_array), 24)
real_data_fore_load_all_arr = remove_nan(real_data_fore_load_all_arr.copy())
 
real_data_fore_total_renewable_all_arr = np.array(real_data_fore_total_renewable_all).reshape(len(dates_array), 24)
real_data_fore_total_renewable_all_arr = remove_nan(real_data_fore_total_renewable_all_arr.copy())

for i in range(len(dates_array)):
    print(i)
    if i + (test_length / 24) > len(dates_array):
        break
 
    
    price_1Y = data_no_nan_price[i].reshape(-1, 24)                    # (num_days_past, 24)
    X_exo_1Y = data_no_nan_fore_load[i].reshape(-1, 24)                # (num_days_past, 24)
    Y_exo_1Y = data_no_nan_fore_total_renewable[i].reshape(-1, 24)     # (num_days_past, 24)
 
    
    X_exo_pred = real_data_fore_load_all_arr[i]                         # (24,)
    Y_exo_pred = real_data_fore_total_renewable_all_arr[i]              # (24,)
 
    
    Train_inp_reg, Train_out_reg, Test_reg = Train_set_Regime(
        price_1Y, X_exo_1Y, Y_exo_1Y,
        X_exo_pred, Y_exo_pred,
        dates_array[i],
    )
 
    XTrain_1Y, XTrain_1Y_raw, YTrain_1Y, XTest_1Y = create_XTrain_YTrain(
        price_1Y, X_exo_1Y, Y_exo_1Y,
        X_exo_pred, Y_exo_pred,
        dates_array[i],
    )
 
    # ── Train/val split ──
    split_idx = int(0.8 * XTrain_1Y.shape[0])
    X_train, y_train = XTrain_1Y[:split_idx, :], YTrain_1Y[:split_idx, :]
    X_val,   y_val   = XTrain_1Y[split_idx:, :], YTrain_1Y[split_idx:, :]
 
    set_seed(123)
    device = get_device()

    fixed = {
        "n_features": X_train.shape[-1],
        "output_dim": y_train.shape[-1],
        "tune_max_epochs": 25,   
        "patience": 6,
        "grad_clip": 1.0,
    }

    # 3) Hyperopt search space
    space = {
        "hidden1": hp.quniform("hidden1", 32, 256, 16),
        "hidden2": hp.quniform("hidden2", 16, 128, 8),
        "num_layers": hp.quniform("num_layers", 1, 5, 1),
        "dropout_p": hp.uniform("dropout_p", 0.05, 0.6),
        "lr": hp.loguniform("lr", math.log(1e-4), math.log(5e-2)),
        "weight_decay": hp.loguniform("weight_decay", math.log(1e-8), math.log(1e-2)),
        "batch_size": hp.choice("batch_size", [32, 64, 128]),
    }

    trials = Trials()

    
    def obj(s):
        return hyperopt_objective(
            space=s,
            fixed=fixed,
            data=(X_train, y_train, X_val, y_val),
            device=device,
            seed=123,
        )

    # 4) Run tuning
    max_evals = 500
    best = fmin(fn=obj, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, rstate=np.random.default_rng(123))
    # Note: hp.choice returns index for batch_size
    batch_size_choices = [32, 64, 128]
    best["batch_size"] = batch_size_choices[int(best["batch_size"])]

    print("\nBest hyperparams (raw):", best)

    
        
    
t.toc()
