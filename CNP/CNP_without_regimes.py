from pytictoc import TicToc
t=TicToc()
import pandas as pd
import numpy as np
from numpy.linalg import cholesky, solve
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
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ds_hdp_hmm_joint_input_output_v2 import*
from merge_regime import*
#from supervised_conditional_neural_process import*
from CNP_V4 import*

warnings.simplefilter('ignore', InterpolationWarning)
warnings.simplefilter("ignore")
Xscaler = StandardScaler()
Yscaler = StandardScaler()

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

def Train_set_Regime(price_vector, X_exo, Y_exo, date):
    XTrain_all = np.zeros((price_vector.shape[0],11))
    YTrain = np.zeros((price_vector.shape[0], 1))
    days = get_days_of_week(date, num_days_past)
    time_points = np.linspace(0,num_days_past,price_vector.shape[0])
    for i in range(7,price_vector.shape[0]-1):
        XTrain_all[i,] = np.concatenate([time_points[i].reshape(1,), np.array([np.mean(price_vector[i-1])]),  np.array([np.mean(price_vector[i-2])]),\
                                      np.array([np.mean(price_vector[i-3])]),  np.array([np.mean(price_vector[i-7])]),  np.array([np.mean(X_exo[i])]),\
                                      np.array([np.mean(X_exo[i-1])]),  np.array([np.mean(X_exo[i-7])]), np.array([np.mean(Y_exo[i])]),\
                                      np.array([np.mean(Y_exo[i-1])]),  np.array([np.mean(Y_exo[i-7])])])
        YTrain[i,] = np.mean(price_vector[i,])
    XTrain_scaled = Xscaler.fit_transform(XTrain_all[7:int(price_vector.shape[0]-1)])
    YTrain_scaled = Yscaler.fit_transform(YTrain[7:int(price_vector.shape[0]-1)])
    XTrain = XTrain_scaled[:-1]
    YTrain = YTrain_scaled[:-1]
    XTest = XTrain_scaled[-1]
    return  XTrain, YTrain, XTest

def create_XTrain_YTrain(price_vector, X_exo, Y_exo, date):
    XTrain_all = np.zeros((price_vector.shape[0],241))
    YTrain_all = np.zeros((price_vector.shape[0],48))
    days = get_days_of_week(date, num_days_past)
    time_points = np.linspace(0,num_days_past,price_vector.shape[0])
    for i in range(7,price_vector.shape[0]-1):
        XTrain_all[i,] = np.concatenate([time_points[i].reshape(1,), price_vector[i-1], price_vector[i-2], price_vector[i-3],
                                     price_vector[i-7], X_exo[i], X_exo[i-1], X_exo[i-7],
                                     Y_exo[i], Y_exo[i-1], Y_exo[i-7]])
        YTrain_all[i,] =  np.concatenate([price_vector[i,], price_vector[i+1,]])
    XTrain_scaled = Xscaler.fit_transform(XTrain_all[7:int(price_vector.shape[0]-1)])
    YTrain_scaled = Yscaler.fit_transform(YTrain_all[7:int(price_vector.shape[0]-1)])
    XTest = np.concatenate([XTrain_scaled[-1].reshape(1,241), days[-1].reshape(1,7)], axis = 1)
    XTrain = np.concatenate([XTrain_scaled[:-1], days[7:int(days.shape[0]-1)]], axis = 1)
    YTrain = YTrain_scaled[:-1]
    XTrain_raw = np.concatenate([XTrain_all[7:int(price_vector.shape[0]-2)], days[7:int(days.shape[0]-1)]], axis = 1)
    return  XTrain, XTrain_raw, YTrain, XTest

    
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
price_data_from_excel = load_data('/PowerAll_updated.xlsx')
fore_gen_data = load_data('Forecasted_generation_Day-Ahead_2015_2024.xlsx')
forecasted_residual_load = load_data('Forecasted_residual_load.xlsx')
forecasted_total_renewable =  pd.DataFrame(fore_gen_data.iloc[9:,3].values.reshape(3378,24))
forecasted_total_renewable.insert(0,'date',forecasted_residual_load.iloc[:,0].values.astype(str).reshape(-1,1),True)

price_data_from_excel = load_data('C:/Users/abhin/Downloads/R-CNP-V2/Datasets/norway_Dataset/norway_aday_ahead_price.xlsx')
forecasted_total_renewable = load_data('C:/Users/abhin/Downloads/R-CNP-V2/Datasets/norway_Dataset/norway_aday_ahead_Total_Generation_Forecast.xlsx')
forecasted_residual_load = load_data('C:/Users/abhin/Downloads/R-CNP-V2/Datasets/norway_Dataset/norway_aday_ahead_Total_Load_Forecast.xlsx')
"""

data_dir = 'C:/Users/abhin/Downloads/R-CNP-V2/Datasets/' + country + '_Dataset/'

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
res_meta = []
final_pred = np.zeros([len(dates_array),48])
final_var = np.zeros([len(dates_array),48])

def fit_cluster_scalers(X: np.ndarray, Y: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
    scalers: Dict[int, Dict[str, np.ndarray]] = {}
    mu_x, std_x = X.mean(axis=0), X.std(axis=0) + 1e-8
    mu_y, std_y = Y.mean(axis=0), Y.std(axis=0) + 1e-8
    scalers = {'mu_x': mu_x, 'std_x': std_x, 'mu_y': mu_y, 'std_y': std_y}
    return scalers


for i in range(len(dates_array)):
    print(i)
    if i+(test_length/24) > len(dates_array):
        break
    ### For GPR
    
    inp_all = np.zeros([data_length, 3])
    real_data_fore_load_all = np.array(real_data_fore_load_all).reshape(len(dates_array),24)
    real_data_fore_load_all = remove_nan(real_data_fore_load_all.copy())

    real_data_fore_total_renewable_all = np.array(real_data_fore_total_renewable_all).reshape(len(dates_array),24)
    real_data_fore_total_renewable_all = remove_nan(real_data_fore_total_renewable_all.copy())

    
    load_all = np.vstack((data_no_nan_fore_load[i],np.array(real_data_fore_load_all[i:i+(int(test_length/24))]).reshape(-1,1)))    
    total_renewable_all = np.vstack((data_no_nan_fore_total_renewable[i],np.array(real_data_fore_total_renewable_all[i:i+(int(test_length/24))]).reshape(-1,1)))
    price_all = np.vstack((data_no_nan_price[i],np.array(real_data_price_all[i:i+(int(test_length/24))]).reshape(-1,1)))

    price_1Y = price_all.reshape(int(price_all.shape[0]/24),24)
    X_exo_1Y = load_all.reshape(int(load_all.shape[0]/24),24)
    Y_exo_1Y = total_renewable_all.reshape(int(total_renewable_all.shape[0]/24),24)
    Train_inp_reg, Train_out_reg, Test_reg = Train_set_Regime(price_1Y, X_exo_1Y, Y_exo_1Y,dates_array[i])
    
    XTrain_1Y, XTrain_1Y_raw, YTrain_1Y, XTest_1Y = create_XTrain_YTrain(price_1Y, X_exo_1Y, Y_exo_1Y,dates_array[i])
    
    
    ids = np.zeros([XTrain_1Y.shape[0],])
    reg_pred = []
    reg_var = []
    res_rep = []
    best_cfg, logs = multitask_successive_halving_search(
            X=remove_nan(XTrain_1Y),            # your training features (all regimes)
            Y=remove_nan(YTrain_1Y),            # your training targets (all regimes)
            cluster_ids=ids,                    # your regime IDs from HDP-HMM merge
            device='cpu',                       # or 'cuda' if available
            n_initial_configs=27,               # ~30 initial configs is plenty
            rung_setups=((20,64,0.4),(80,96,0.7),(200,128,1.0))  # tweak to budget
        )
    print("Best shared hyper-params:", best_cfg)
    for rep in range(10):
        model = train_cnp_single_cluster(
            remove_nan(XTrain_1Y), remove_nan(YTrain_1Y), hidden_dim=best_cfg['hidden_dim'],
            num_layers=best_cfg['num_layers'], activation=best_cfg['activation'],
            dropout=best_cfg['dropout'], lr=best_cfg['lr'], weight_decay=best_cfg['weight_decay'],
            epochs = 1000, batch_size = 128, min_ctx = 16, ctx_frac = 0.5, pivots = best_cfg['pivots'],
            grad_clip = 1.0, val_frac = 0.2, patience = 20, device = 'cpu'
        )
        models = {}
        scalers = {}
        models[0] = model
        scaler = fit_cluster_scalers(remove_nan(XTrain_1Y), remove_nan(YTrain_1Y))
        scalers[0] = scaler
        x_star_raw = XTest_1Y
        res = predict_single_target_mixture(
            x_star_raw, XTrain_1Y, YTrain_1Y, ids, models, scalers, k_ctx=1, device='cpu'
        )
        aa = Yscaler.inverse_transform(res['mean_mix'].reshape(-1,48))
        bb = Yscaler.inverse_transform(res['var_mix'].reshape(-1,48))
        reg_pred.append(aa)
        reg_var.append(bb)
    final_pred[i,] = np.mean(np.asarray(reg_pred), axis = 0)
    final_var[i,] = np.mean(np.asarray(reg_var), axis = 0)
    save_pred_res_as = 'pred_' + country + '_no_regimes_' + dates_array[i] + '.txt'
    save_var_res_as = 'var_' + country + '_no_regimes_' + dates_array[i] + '.txt'
    np.savetxt(save_pred_res_as, final_pred)
    np.savetxt(save_var_res_as, final_var)
   
t.toc()
