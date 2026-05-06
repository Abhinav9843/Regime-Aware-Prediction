import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.utils._testing import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import argparse
import os

warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
Xscaler = StandardScaler()
Yscaler = StandardScaler()

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

# Example of synthetic data generation
np.random.seed(42)



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
            
def get_num_day_future(prompt):
    while True:
        try:
            input_val = input(prompt + " (Integer Value): ")
            num_days = int(input_val)
            return num_days
        except ValueError:
            print("Invalid value entered. Please enter a integer value.")

def find_day_of_week(year, month, day):
    date_object = datetime(year, month, day)
    day_of_week = date_object.strftime("%A")
    return day_of_week


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
"""
def create_XTrain_YTrain(price_vector, X_exo, Y_exo):
    XTrain = np.zeros((price_vector.shape[0],240))
    YTrain = np.zeros((price_vector.shape[0],24))
    for i in range(6,price_vector.shape[0]):
        XTrain[i,] = np.concatenate([price_vector[i-1], price_vector[i-2], price_vector[i-3],
                                     price_vector[i-7], X_exo[i], X_exo[i-1], X_exo[i-7],
                                     Y_exo[i], Y_exo[i-1], Y_exo[i-7]])
        YTrain[i,] =  price_vector[i,]
    #print(XTest)
    #XTest = XTrain[-1:]
    XTrain_scaled = Xscaler.fit_transform(XTrain[6:int(XTrain.shape[0])])
    YTrain_scaled = Yscaler.fit_transform(YTrain[6:int(YTrain.shape[0])])
    XTest = XTrain_scaled[-1:]
    XTrain = XTrain_scaled[6:int(XTrain_scaled.shape[0]-1)]
    YTrain = YTrain_scaled[6:int(YTrain_scaled.shape[0]-1)]
    return  XTrain, YTrain, XTest
"""
def create_XTrain_YTrain(price_vector, X_exo, Y_exo, date):
    XTrain_all = np.zeros((price_vector.shape[0],241))
    YTrain_all = np.zeros((price_vector.shape[0],24))
    days = get_days_of_week(date, num_days_past)
    time_points = np.linspace(0,num_days_past,price_vector.shape[0])
    for i in range(7,price_vector.shape[0]-1):
        XTrain_all[i,] = np.concatenate([time_points[i].reshape(1,), price_vector[i-1], price_vector[i-2], price_vector[i-3],
                                     price_vector[i-7], X_exo[i], X_exo[i-1], X_exo[i-7],
                                     Y_exo[i], Y_exo[i-1], Y_exo[i-7]])
        YTrain_all[i,] =  price_vector[i,]#np.concatenate([price_vector[i,], price_vector[i+1,]])
    XTrain_scaled = Xscaler.fit_transform(XTrain_all[7:int(price_vector.shape[0]-1)])
    YTrain_scaled = Yscaler.fit_transform(YTrain_all[7:int(price_vector.shape[0]-1)])
    XTest = np.concatenate([XTrain_scaled[-1].reshape(1,241), days[-1].reshape(1,7)], axis = 1)
    XTrain = np.concatenate([XTrain_scaled[:-1], days[7:int(days.shape[0]-1)]], axis = 1)
    YTrain = YTrain_scaled[:-1]
    XTrain_raw = np.concatenate([XTrain_all[7:int(price_vector.shape[0]-2)], days[7:int(days.shape[0]-1)]], axis = 1)
    return  XTrain, YTrain, XTest

def create_data_set(data_from_excel,dates_array):
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
            data_no_nan_all.append(data_no_nan.reshape(-1,1))
            
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
            data_no_nan_all.append(data_no_nan.reshape(-1,1))
    return data_all, mn_all , sd_all, real_data_all, data_no_nan_all
 #################################################################################
 ###############################################################################
country = str(input("Enter Country:\t"))
start_date = get_date_input("Enter the start date")
end_date = get_date_input("Enter the end date")
dates_array = create_date_array(start_date, end_date)
num_days_past = int(input("Enter Number of Days for Training: "))

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

data_all_price, mn_all, sd_all,real_data_price_all,data_no_nan_price = create_data_set(price_data_from_excel,dates_array)
data_all_fore_load,_,_,real_data_fore_load_all,data_no_nan_fore_load = create_data_set(forecasted_residual_load,dates_array)
data_all_fore_total_renewable,_,_,real_data_fore_total_renewable_all,data_no_nan_fore_total_renewable = create_data_set(forecasted_total_renewable,dates_array)

test_length = 24
data_length = data_no_nan_price[0].shape[0]+test_length
train_length = data_length - test_length


pred_LEAR_4Y = np.zeros([len(dates_array),24])
pred_LEAR_3Y = np.zeros([len(dates_array),24])
pred_LEAR_8W = np.zeros([len(dates_array),24])
pred_LEAR_12W = np.zeros([len(dates_array),24])
Final_pred_LEAR = np.zeros([len(dates_array),24])

mae_error_gpr = np.zeros([len(dates_array),1])
mae_error_LEAR = np.zeros([len(dates_array),1])

for i in range(len(dates_array)):
    print(i)
    if i+(test_length/24) > len(dates_array):
        break
    inp_all = np.zeros([data_length, 3])
    real_data_fore_load_all = np.array(real_data_fore_load_all).reshape(len(dates_array),24)
    real_data_fore_load_all = remove_nan(real_data_fore_load_all.copy())

    real_data_fore_total_renewable_all = np.array(real_data_fore_total_renewable_all).reshape(len(dates_array),24)
    real_data_fore_total_renewable_all = remove_nan(real_data_fore_total_renewable_all.copy())

    
    load_all = np.vstack((data_no_nan_fore_load[i],np.array(real_data_fore_load_all[i:i+(int(test_length/24))]).reshape(-1,1)))    
    total_renewable_all = np.vstack((data_no_nan_fore_total_renewable[i],np.array(real_data_fore_total_renewable_all[i:i+(int(test_length/24))]).reshape(-1,1)))
    price_all = np.vstack((data_no_nan_price[i],np.array(real_data_price_all[i:i+(int(test_length/24))]).reshape(-1,1)))

    price_4Y = price_all.reshape(int(price_all.shape[0]/24),24)
    X_exo_4Y = load_all.reshape(int(load_all.shape[0]/24),24)
    Y_exo_4Y = total_renewable_all.reshape(int(total_renewable_all.shape[0]/24),24)
    XTrain_4Y, YTrain_4Y, XTest_4Y = create_XTrain_YTrain(price_4Y, X_exo_4Y, Y_exo_4Y, dates_array[i])

    price_3Y = price_4Y[365:]
    X_exo_3Y = X_exo_4Y[365:]
    Y_exo_3Y = Y_exo_4Y[365:]
    
    param_model = LassoLarsIC(criterion='aic', max_iter=2500)
    pred_4Y = np.zeros([24,1])
    pred_3Y = np.zeros([24,1])
    for h in range(24):
        param_4Y = param_model.fit(XTrain_4Y, YTrain_4Y[:,h]).alpha_
        model_4Y = Lasso(max_iter = 2500, alpha = param_4Y)
        model_4Y.fit(XTrain_4Y, YTrain_4Y[:,h])

        
        
        pred_4Y[h] = model_4Y.predict(XTest_4Y)
        
    pred_4Y = Yscaler.inverse_transform(pred_4Y.reshape(1,24))
    
    pred_LEAR_4Y[i,] = pred_4Y
    
    Final_pred_LEAR[i,] = pred_4Y

    mae_error_LEAR[i] = np.mean(np.abs(Final_pred_LEAR[i,] - price_4Y[-1]))
    print(mae_error_LEAR[i])
    
    
    
        
    
