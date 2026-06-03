from trading_strategies_I_IV import*



country_list = ["germany", "france", "norway"]

for idx, cnt in enumerate(country_list):
    pred_cnp = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/Result_' + \
                          cnt + '_tuned_new/'+ 'pred_cnp_' + cnt + '.txt')
    pred_cnp = pred_cnp[:,:24]
    pred_cnp_no_regimes = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/CNP_no_regimes/pred_' + \
                                     cnt + '_no_regimes_new.txt')
    pred_cnp_no_regimes = pred_cnp_no_regimes[:,:24]
    pred_xgb = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/XGB/pred_xbg_' + cnt + '_2023' + '.txt')
    pred_xgb = pred_xgb[:,:24]
    var_xgb = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/XGB/var_xgb_' + cnt + '.txt')[:365,:24]
    pred_dnn = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/DNN/DNN_' + cnt + '_2023' + '.txt')
    var_dnn = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/DNN/var_dnn_' + cnt + '.txt')[:365,:24]
    pred_lear = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/LEAR/Final_pred_LEAR_' + cnt + '.txt')
    var_lear = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/LEAR/var_lear_' + cnt + '.txt')[:365,:24]
    pred_blstm = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/BLSTM/pred_blstm_' + cnt + '.txt')
    pred_tft = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/TFT/pred_tft_' + cnt + '.txt')[:365,:24]
    var_tft = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/TFT/var_tft_' + cnt + '.txt')[:365,:24]

    var_cnp = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/Result_' + \
                          cnt + '_tuned_new/'+ 'var_cnp_' + cnt + '.txt')
    var_cnp = var_cnp[:,:24]

    var_cnp_no_regimes =  np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/CNP_no_regimes/var_' + \
                                     cnt + '_no_regimes_new.txt')
    var_cnp_no_regimes = var_cnp_no_regimes[:,:24]
    var_blstm = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/BLSTM/var_blstm_' + cnt + '.txt')

    real = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/real_data/real_'+cnt+'.txt')
    load = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_load_data_'+cnt+'.txt')
    solar = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_pv_data_'+cnt+'.txt')

    
    profit_cnp_case_I = np.zeros([real.shape[0]])
    profit_cnp_case_II = np.zeros([real.shape[0]])
    profit_cnp_case_III = np.zeros([real.shape[0]])
    cost_cnp_case_IV = np.zeros([real.shape[0]])
    perfect_foresight_cnp_case_I = np.zeros([real.shape[0]])
    perfect_foresight_cnp_case_II = np.zeros([real.shape[0]])
    perfect_foresight_cnp_case_III = np.zeros([real.shape[0]])
    perfect_foresight_cnp_case_IV = np.zeros([real.shape[0]])

    profit_tft_case_I = np.zeros([real.shape[0]])
    profit_tft_case_II = np.zeros([real.shape[0]])
    profit_tft_case_III = np.zeros([real.shape[0]])
    cost_tft_case_IV = np.zeros([real.shape[0]])
    perfect_foresight_tft_case_I = np.zeros([real.shape[0]])
    perfect_foresight_tft_case_II = np.zeros([real.shape[0]])
    perfect_foresight_tft_case_III = np.zeros([real.shape[0]])
    perfect_foresight_tft_case_IV = np.zeros([real.shape[0]])

    profit_lear_case_I = np.zeros([real.shape[0]])
    profit_lear_case_II = np.zeros([real.shape[0]])
    profit_lear_case_III = np.zeros([real.shape[0]])
    cost_lear_case_IV = np.zeros([real.shape[0]])
    perfect_foresight_lear_case_I = np.zeros([real.shape[0]])
    perfect_foresight_lear_case_II = np.zeros([real.shape[0]])
    perfect_foresight_lear_case_III = np.zeros([real.shape[0]])
    perfect_foresight_lear_case_IV = np.zeros([real.shape[0]])

    profit_xgb_case_I = np.zeros([real.shape[0]])
    profit_xgb_case_II = np.zeros([real.shape[0]])
    profit_xgb_case_III = np.zeros([real.shape[0]])
    cost_xgb_case_IV = np.zeros([real.shape[0]])
    perfect_foresight_xgb_case_I = np.zeros([real.shape[0]])
    perfect_foresight_xgb_case_II = np.zeros([real.shape[0]])
    perfect_foresight_xgb_case_III = np.zeros([real.shape[0]])
    perfect_foresight_xgb_case_IV = np.zeros([real.shape[0]])

    profit_dnn_case_I = np.zeros([real.shape[0]])
    profit_dnn_case_II = np.zeros([real.shape[0]])
    profit_dnn_case_III = np.zeros([real.shape[0]])
    cost_dnn_case_IV = np.zeros([real.shape[0]])
    perfect_foresight_dnn_case_I = np.zeros([real.shape[0]])
    perfect_foresight_dnn_case_II = np.zeros([real.shape[0]])
    perfect_foresight_dnn_case_III = np.zeros([real.shape[0]])
    perfect_foresight_dnn_case_IV = np.zeros([real.shape[0]])

    profit_blstm_case_I = np.zeros([real.shape[0]])
    profit_blstm_case_II = np.zeros([real.shape[0]])
    profit_blstm_case_III = np.zeros([real.shape[0]])
    cost_blstm_case_IV = np.zeros([real.shape[0]])
    perfect_foresight_blstm_case_I = np.zeros([real.shape[0]])
    perfect_foresight_blstm_case_II = np.zeros([real.shape[0]])
    perfect_foresight_blstm_case_III = np.zeros([real.shape[0]])
    perfect_foresight_blstm_case_IV = np.zeros([real.shape[0]])

    profit_cnp_no_regimes_case_I = np.zeros([real.shape[0]])
    profit_cnp_no_regimes_case_II = np.zeros([real.shape[0]])
    profit_cnp_no_regimes_case_III = np.zeros([real.shape[0]])
    cost_cnp_no_regimes_case_IV = np.zeros([real.shape[0]])
    perfect_foresight_cnp_no_regimes_case_I = np.zeros([real.shape[0]])
    perfect_foresight_cnp_no_regimes_case_II = np.zeros([real.shape[0]])
    perfect_foresight_cnp_no_regimes_case_III = np.zeros([real.shape[0]])
    perfect_foresight_cnp_no_regimes_case_IV = np.zeros([real.shape[0]])



    
    for days in range(real.shape[0]):
        realized_prices = real[days]
        realized_load = load[days]
        realized_renewable_availability = solar[days]

        #### CNP
        
        profit_cnp_case_I[days], _, _, _ = solve_case_i(pred_cnp[days], realized_prices, is_perfect_foresight=False)
        perfect_foresight_cnp_case_I[days], _, _, _ = solve_case_i(real[days], realized_prices, is_perfect_foresight=True)

        profit_cnp_case_II[days], _, _, _ = solve_case_ii(pred_cnp[days], realized_prices, np.sqrt(var_cnp[days]), is_perfect_foresight=False)
        perfect_foresight_cnp_case_II[days], _, _, _ = solve_case_ii(real[days], realized_prices, np.sqrt(var_cnp[days]), is_perfect_foresight=True)

        profit_cnp_case_III[days], _, _, _ = solve_case_iii(pred_cnp[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp[days]), load[days], solar[days], is_perfect_foresight=False)
        perfect_foresight_cnp_case_III[days], _, _, _ = solve_case_iii(real[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp[days]), load[days], solar[days], is_perfect_foresight=True)

        cost_cnp_case_IV[days], _, _, _, _ = solve_case_iv(pred_cnp[days], realized_prices, np.sqrt(var_cnp[days]), fixed_load, is_perfect_foresight=False)
        perfect_foresight_cnp_case_IV[days], _, _, _, _ = solve_case_iv(real[days], realized_prices, np.sqrt(var_cnp[days]), fixed_load, is_perfect_foresight=True)


        #### XGB

        profit_xgb_case_I[days], _, _, _ = solve_case_i(pred_xgb[days], realized_prices, is_perfect_foresight=False)
        perfect_foresight_xgb_case_I[days], _, _, _ = solve_case_i(real[days], realized_prices, is_perfect_foresight=True)

        profit_xgb_case_II[days], _, _, _ = solve_case_ii(pred_xgb[days], realized_prices, np.sqrt(var_xgb[days]), is_perfect_foresight=False)
        perfect_foresight_xgb_case_II[days], _, _, _ = solve_case_ii(real[days], realized_prices, np.sqrt(var_xgb[days]), is_perfect_foresight=True)

        profit_xgb_case_III[days], _, _, _ = solve_case_iii(pred_xgb[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_xgb[days]), load[days], solar[days], is_perfect_foresight=False)
        perfect_foresight_xgb_case_III[days], _, _, _ = solve_case_iii(real[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_xgb[days]), load[days], solar[days], is_perfect_foresight=True)

        cost_xgb_case_IV[days], _, _, _, _ = solve_case_iv(pred_xgb[days], realized_prices, np.sqrt(var_xgb[days]), fixed_load, is_perfect_foresight=False)
        perfect_foresight_xgb_case_IV[days], _, _, _, _ = solve_case_iv(real[days], realized_prices, np.sqrt(var_xgb[days]), fixed_load, is_perfect_foresight=True)

        #### LEAR

        profit_lear_case_I[days], _, _, _ = solve_case_i(pred_lear[days], realized_prices, is_perfect_foresight=False)
        perfect_foresight_lear_case_I[days], _, _, _ = solve_case_i(real[days], realized_prices, is_perfect_foresight=True)

        profit_lear_case_II[days], _, _, _ = solve_case_ii(pred_lear[days], realized_prices, np.sqrt(var_lear[days]), is_perfect_foresight=False)
        perfect_foresight_lear_case_II[days], _, _, _ = solve_case_ii(real[days], realized_prices, np.sqrt(var_lear[days]), is_perfect_foresight=True)

        profit_lear_case_III[days], _, _, _ = solve_case_iii(pred_lear[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_lear[days]), load[days], solar[days], is_perfect_foresight=False)
        perfect_foresight_lear_case_III[days], _, _, _ = solve_case_iii(real[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_lear[days]), load[days], solar[days], is_perfect_foresight=True)

        cost_lear_case_IV[days], _, _, _, _ = solve_case_iv(pred_lear[days], realized_prices, np.sqrt(var_lear[days]), fixed_load, is_perfect_foresight=False)
        perfect_foresight_lear_case_IV[days], _, _, _, _ = solve_case_iv(real[days], realized_prices, np.sqrt(var_lear[days]), fixed_load, is_perfect_foresight=True)

        ##### DNN
        
        profit_dnn_case_I[days], _, _, _ = solve_case_i(pred_dnn[days], realized_prices, is_perfect_foresight=False)
        perfect_foresight_dnn_case_I[days], _, _, _ = solve_case_i(real[days], realized_prices, is_perfect_foresight=True)

        profit_dnn_case_II[days], _, _, _ = solve_case_ii(pred_dnn[days], realized_prices, np.sqrt(var_dnn[days]), is_perfect_foresight=False)
        perfect_foresight_dnn_case_II[days], _, _, _ = solve_case_ii(real[days], realized_prices, np.sqrt(var_dnn[days]), is_perfect_foresight=True)

        profit_dnn_case_III[days], _, _, _ = solve_case_iii(pred_dnn[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_dnn[days]), load[days], solar[days], is_perfect_foresight=False)
        perfect_foresight_dnn_case_III[days], _, _, _ = solve_case_iii(real[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_dnn[days]), load[days], solar[days], is_perfect_foresight=True)

        cost_dnn_case_IV[days], _, _, _, _ = solve_case_iv(pred_dnn[days], realized_prices, np.sqrt(var_dnn[days]), fixed_load, is_perfect_foresight=False)
        perfect_foresight_dnn_case_IV[days], _, _, _, _ = solve_case_iv(real[days], realized_prices, np.sqrt(var_dnn[days]), fixed_load, is_perfect_foresight=True)

        ##### BLSTM

        profit_blstm_case_I[days], _, _, _ = solve_case_i(pred_blstm[days], realized_prices, is_perfect_foresight=False)
        perfect_foresight_blstm_case_I[days], _, _, _ = solve_case_i(real[days], realized_prices, is_perfect_foresight=True)

        profit_blstm_case_II[days], _, _, _ = solve_case_ii(pred_blstm[days], realized_prices, np.sqrt(var_blstm[days]), is_perfect_foresight=False)
        perfect_foresight_blstm_case_II[days], _, _, _ = solve_case_ii(real[days], realized_prices, np.sqrt(var_blstm[days]), is_perfect_foresight=True)

        profit_blstm_case_III[days], _, _, _ = solve_case_iii(pred_blstm[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_blstm[days]), load[days], solar[days], is_perfect_foresight=False)
        perfect_foresight_blstm_case_III[days], _, _, _ = solve_case_iii(real[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_blstm[days]), load[days], solar[days], is_perfect_foresight=True)

        cost_blstm_case_IV[days], _, _, _, _ = solve_case_iv(pred_blstm[days], realized_prices, np.sqrt(var_blstm[days]), fixed_load, is_perfect_foresight=False)
        perfect_foresight_blstm_case_IV[days], _, _, _, _ = solve_case_iv(real[days], realized_prices, np.sqrt(var_blstm[days]), fixed_load, is_perfect_foresight=True)

        ##### TFT

        profit_tft_case_I[days], _, _, _ = solve_case_i(pred_tft[days], realized_prices, is_perfect_foresight=False)
        perfect_foresight_tft_case_I[days], _, _, _ = solve_case_i(real[days], realized_prices, is_perfect_foresight=True)

        profit_tft_case_II[days], _, _, _ = solve_case_ii(pred_tft[days], realized_prices, np.sqrt(var_tft[days]), is_perfect_foresight=False)
        perfect_foresight_tft_case_II[days], _, _, _ = solve_case_ii(real[days], realized_prices, np.sqrt(var_tft[days]), is_perfect_foresight=True)

        profit_tft_case_III[days], _, _, _ = solve_case_iii(pred_tft[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_tft[days]), load[days], solar[days], is_perfect_foresight=False)
        perfect_foresight_tft_case_III[days], _, _, _ = solve_case_iii(real[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_tft[days]), load[days], solar[days], is_perfect_foresight=True)

        cost_tft_case_IV[days], _, _, _, _ = solve_case_iv(pred_tft[days], realized_prices, np.sqrt(var_tft[days]), fixed_load, is_perfect_foresight=False)
        perfect_foresight_tft_case_IV[days], _, _, _, _ = solve_case_iv(real[days], realized_prices, np.sqrt(var_tft[days]), fixed_load, is_perfect_foresight=True)

        #### CNP_NO_Regimes

        profit_cnp_no_regimes_case_I[days], _, _, _ = solve_case_i(pred_cnp_no_regimes[days], realized_prices, is_perfect_foresight=False)
        perfect_foresight_cnp_no_regimes_case_I[days], _, _, _ = solve_case_i(real[days], realized_prices, is_perfect_foresight=True)

        profit_cnp_no_regimes_case_II[days], _, _, _ = solve_case_ii(pred_cnp_no_regimes[days], realized_prices, np.sqrt(var_cnp_no_regimes[days]), is_perfect_foresight=False)
        perfect_foresight_cnp_no_regimes_case_II[days], _, _, _ = solve_case_ii(real[days], realized_prices, np.sqrt(var_cnp_no_regimes[days]), is_perfect_foresight=True)

        profit_cnp_no_regimes_case_III[days], _, _, _ = solve_case_iii(pred_cnp_no_regimes[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_no_regimes[days]), load[days], solar[days], is_perfect_foresight=False)
        perfect_foresight_cnp_no_regimes_case_III[days], _, _, _ = solve_case_iii(real[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_no_regimes[days]), load[days], solar[days], is_perfect_foresight=True)

        cost_cnp_no_regimes_case_IV[days], _, _, _, _ = solve_case_iv(pred_cnp_no_regimes[days], realized_prices, np.sqrt(var_cnp_no_regimes[days]), fixed_load, is_perfect_foresight=False)
        perfect_foresight_cnp_no_regimes_case_IV[days], _, _, _, _ = solve_case_iv(real[days], realized_prices, np.sqrt(var_cnp_no_regimes[days]), fixed_load, is_perfect_foresight=True)


    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'profit_cnp_case_I.txt', profit_cnp_case_I)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'profit_cnp_case_II.txt', profit_cnp_case_II)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'profit_cnp_case_III.txt', profit_cnp_case_III)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'cost_cnp_case_IV.txt', cost_cnp_case_IV)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_case_I.txt', perfect_foresight_cnp_case_I)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_case_II.txt', perfect_foresight_cnp_case_II)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_case_III.txt', perfect_foresight_cnp_case_III)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_case_IV.txt', perfect_foresight_cnp_case_IV)

    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'profit_tft_case_I.txt', profit_tft_case_I)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'profit_tft_case_II.txt', profit_tft_case_II)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'profit_tft_case_III.txt', profit_tft_case_III)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'cost_tft_case_IV.txt', cost_tft_case_IV)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_case_I.txt', perfect_foresight_tft_case_I)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_case_II.txt', perfect_foresight_tft_case_II)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_case_III.txt', perfect_foresight_tft_case_III)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_case_IV.txt', perfect_foresight_tft_case_IV)


    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'profit_dnn_case_I.txt', profit_dnn_case_I)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'profit_dnn_case_II.txt', profit_dnn_case_II)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'profit_dnn_case_III.txt', profit_dnn_case_III)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'cost_dnn_case_IV.txt', cost_dnn_case_IV)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_case_I.txt', perfect_foresight_dnn_case_I)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_case_II.txt', perfect_foresight_dnn_case_II)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_case_III.txt', perfect_foresight_dnn_case_III)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_case_IV.txt', perfect_foresight_dnn_case_IV)


    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'profit_lear_case_I.txt', profit_lear_case_I)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'profit_lear_case_II.txt', profit_lear_case_II)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'profit_lear_case_III.txt', profit_lear_case_III)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'cost_lear_case_IV.txt', cost_lear_case_IV)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_case_I.txt', perfect_foresight_lear_case_I)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_case_II.txt', perfect_foresight_lear_case_II)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_case_III.txt', perfect_foresight_lear_case_III)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_case_IV.txt', perfect_foresight_lear_case_IV)


    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'profit_blstm_case_I.txt', profit_blstm_case_I)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'profit_blstm_case_II.txt', profit_blstm_case_II)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'profit_blstm_case_III.txt', profit_blstm_case_III)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'cost_blstm_case_IV.txt', cost_blstm_case_IV)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_case_I.txt', perfect_foresight_blstm_case_I)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_case_II.txt', perfect_foresight_blstm_case_II)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_case_III.txt', perfect_foresight_blstm_case_III)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_case_IV.txt', perfect_foresight_blstm_case_IV)


    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'profit_xgb_case_I.txt', profit_xgb_case_I)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'profit_xgb_case_II.txt', profit_xgb_case_II)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'profit_xgb_case_III.txt', profit_xgb_case_III)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'cost_xgb_case_IV.txt', cost_xgb_case_IV)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_case_I.txt', perfect_foresight_xgb_case_I)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_case_II.txt', perfect_foresight_xgb_case_II)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_case_III.txt', perfect_foresight_xgb_case_III)
    np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_case_IV.txt', perfect_foresight_xgb_case_IV)


    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'profit_cnp_no_regime_case_I.txt', profit_cnp_no_regimes_case_I)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'profit_cnp_no_regime_case_II.txt', profit_cnp_no_regimes_case_II)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'profit_cnp_no_regime_case_III.txt', profit_cnp_no_regimes_case_III)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'cost_cnp_no_regime_case_IV.txt', cost_cnp_no_regimes_case_IV)

    






    if cnt == 'germany':
        pred_cnp_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/Result_germany_tuned_2021/pred_cnp_germany_2021.txt')[:365,:24]
        var_cnp_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/Result_germany_tuned_2021/var_cnp_germany_2021.txt')[:365,:24]
        pred_cnp_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/Result_germany_tuned_2022/pred_cnp_germany_2022.txt')[:365,:24]
        var_cnp_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/Result_germany_tuned_2022/var_cnp_germany_2022.txt')[:365,:24]
        
        pred_blstm_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/BLSTM/pred_blstm_germany_2021.txt')[:365,:24]
        pred_blstm_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/BLSTM/pred_blstm_germany_2022.txt')[:365,:24]
        var_blstm_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/BLSTM/var_blstm_germany_2021.txt')[:365,:24]
        var_blstm_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/BLSTM/var_blstm_germany_2022.txt')[:365,:24]

        pred_lear_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/LEAR/Final_pred_LEAR_Germany_2021.txt')[:365,:24]
        pred_lear_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/LEAR/Final_pred_LEAR_Germany_2022.txt')[:365,:24]
        var_lear_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/LEAR/var_lear_2021_' + cnt + '.txt')
        var_lear_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/LEAR/var_lear_2022_' + cnt + '.txt')

        pred_dnn_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/DNN/DNN_germany_2021.txt')[:365,:24]
        pred_dnn_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/DNN/DNN_germany_2022.txt')[:365,:24]
        var_dnn_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/DNN/var_dnn_2021_' + cnt + '.txt')[:365,:24]
        var_dnn_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/DNN/var_dnn_2022_' + cnt + '.txt')[:365,:24]

        pred_tft_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/TFT/pred_tft_germany_2021.txt')[:365,:24]
        pred_tft_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/TFT/pred_tft_germany_2022.txt')[:365,:24]
        var_tft_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/TFT/var_tft_2021_' + cnt + '.txt')[:365,:24]
        var_tft_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/TFT/var_tft_2022_' + cnt + '.txt')[:365,:24]

        pred_xgb_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/XGB/pred_xgb_germany_2021.txt')[:365,:24]
        pred_xgb_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/XGB/pred_xgb_germany_2022.txt')[:365,:24]
        var_xgb_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/XGB/var_xgb_2021_' + cnt + '.txt')[:365,:24]
        var_xgb_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/XGB/var_xgb_2022_' + cnt + '.txt')[:365,:24]

        
        pred_cnp_nr_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/CNP_no_regimes/pred_germany_no_regimes_2021.txt')[:365,:24]
        var_cnp_nr_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/CNP_no_regimes/var_germany_no_regimes_2021_new.txt')[:365,:24]
        pred_cnp_nr_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/CNP_no_regimes/pred_germany_no_regimes_2022.txt')[:365,:24]
        var_cnp_nr_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/CNP_no_regimes/var_germany_no_regimes_2022_new.txt')[:365,:24]

        real_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/real_data/real_germany_2021.txt')[:365,:24]
        real_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/real_data/real_germany_2022.txt')[:365,:24]
        load_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_load_data_'+cnt+'_2021.txt')[:365,:24]
        solar_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_pv_data_'+cnt+'_2021.txt')[:365,:24]
        load_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_load_data_'+cnt+'_2022.txt')[:365,:24]
        solar_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_pv_data_'+cnt+'_2022.txt')[:365,:24]

        profit_cnp_2021_case_I = np.zeros([real.shape[0]])
        profit_cnp_2021_case_II = np.zeros([real.shape[0]])
        profit_cnp_2021_case_III = np.zeros([real.shape[0]])
        cost_cnp_2021_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_cnp_2021_case_I = np.zeros([real.shape[0]])
        perfect_foresight_cnp_2021_case_II = np.zeros([real.shape[0]])
        perfect_foresight_cnp_2021_case_III = np.zeros([real.shape[0]])
        perfect_foresight_cnp_2021_case_IV = np.zeros([real.shape[0]])

        profit_blstm_2021_case_I = np.zeros([real.shape[0]])
        profit_blstm_2021_case_II = np.zeros([real.shape[0]])
        profit_blstm_2021_case_III = np.zeros([real.shape[0]])
        cost_blstm_2021_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_blstm_2021_case_I = np.zeros([real.shape[0]])
        perfect_foresight_blstm_2021_case_II = np.zeros([real.shape[0]])
        perfect_foresight_blstm_2021_case_III = np.zeros([real.shape[0]])
        perfect_foresight_blstm_2021_case_IV = np.zeros([real.shape[0]])

        profit_lear_2021_case_I = np.zeros([real.shape[0]])
        profit_lear_2021_case_II = np.zeros([real.shape[0]])
        profit_lear_2021_case_III = np.zeros([real.shape[0]])
        cost_lear_2021_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_lear_2021_case_I = np.zeros([real.shape[0]])
        perfect_foresight_lear_2021_case_II = np.zeros([real.shape[0]])
        perfect_foresight_lear_2021_case_III = np.zeros([real.shape[0]])
        perfect_foresight_lear_2021_case_IV = np.zeros([real.shape[0]])

        profit_dnn_2021_case_I = np.zeros([real.shape[0]-3])
        profit_dnn_2021_case_II = np.zeros([real.shape[0]-3])
        profit_dnn_2021_case_III = np.zeros([real.shape[0]-3])
        cost_dnn_2021_case_IV = np.zeros([real.shape[0]-3])
        perfect_foresight_dnn_2021_case_I = np.zeros([real.shape[0]-3])
        perfect_foresight_dnn_2021_case_II = np.zeros([real.shape[0]-3])
        perfect_foresight_dnn_2021_case_III = np.zeros([real.shape[0]-3])
        perfect_foresight_dnn_2021_case_IV = np.zeros([real.shape[0]-3])

        profit_xgb_2021_case_I = np.zeros([real.shape[0]])
        profit_xgb_2021_case_II = np.zeros([real.shape[0]])
        profit_xgb_2021_case_III = np.zeros([real.shape[0]])
        cost_xgb_2021_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_xgb_2021_case_I = np.zeros([real.shape[0]])
        perfect_foresight_xgb_2021_case_II = np.zeros([real.shape[0]])
        perfect_foresight_xgb_2021_case_III = np.zeros([real.shape[0]])
        perfect_foresight_xgb_2021_case_IV = np.zeros([real.shape[0]])

        profit_tft_2021_case_I = np.zeros([real.shape[0]])
        profit_tft_2021_case_II = np.zeros([real.shape[0]])
        profit_tft_2021_case_III = np.zeros([real.shape[0]])
        cost_tft_2021_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_tft_2021_case_I = np.zeros([real.shape[0]])
        perfect_foresight_tft_2021_case_II = np.zeros([real.shape[0]])
        perfect_foresight_tft_2021_case_III = np.zeros([real.shape[0]])
        perfect_foresight_tft_2021_case_IV = np.zeros([real.shape[0]])

        profit_cnp_nr_2021_case_I = np.zeros([real.shape[0]])
        profit_cnp_nr_2021_case_II = np.zeros([real.shape[0]])
        profit_cnp_nr_2021_case_III = np.zeros([real.shape[0]])
        cost_cnp_nr_2021_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_cnp_nr_2021_case_I = np.zeros([real.shape[0]])
        perfect_foresight_cnp_nr_2021_case_II = np.zeros([real.shape[0]])
        perfect_foresight_cnp_nr_2021_case_III = np.zeros([real.shape[0]])
        perfect_foresight_cnp_nr_2021_case_IV = np.zeros([real.shape[0]])

        profit_cnp_2022_case_I = np.zeros([real.shape[0]])
        profit_cnp_2022_case_II = np.zeros([real.shape[0]])
        profit_cnp_2022_case_III = np.zeros([real.shape[0]])
        cost_cnp_2022_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_cnp_2022_case_I = np.zeros([real.shape[0]])
        perfect_foresight_cnp_2022_case_II = np.zeros([real.shape[0]])
        perfect_foresight_cnp_2022_case_III = np.zeros([real.shape[0]])
        perfect_foresight_cnp_2022_case_IV = np.zeros([real.shape[0]])

        profit_blstm_2022_case_I = np.zeros([real.shape[0]])
        profit_blstm_2022_case_II = np.zeros([real.shape[0]])
        profit_blstm_2022_case_III = np.zeros([real.shape[0]])
        cost_blstm_2022_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_blstm_2022_case_I = np.zeros([real.shape[0]])
        perfect_foresight_blstm_2022_case_II = np.zeros([real.shape[0]])
        perfect_foresight_blstm_2022_case_III = np.zeros([real.shape[0]])
        perfect_foresight_blstm_2022_case_IV = np.zeros([real.shape[0]])

        profit_lear_2022_case_I = np.zeros([real.shape[0]])
        profit_lear_2022_case_II = np.zeros([real.shape[0]])
        profit_lear_2022_case_III = np.zeros([real.shape[0]])
        cost_lear_2022_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_lear_2022_case_I = np.zeros([real.shape[0]])
        perfect_foresight_lear_2022_case_II = np.zeros([real.shape[0]])
        perfect_foresight_lear_2022_case_III = np.zeros([real.shape[0]])
        perfect_foresight_lear_2022_case_IV = np.zeros([real.shape[0]])

        profit_dnn_2022_case_I = np.zeros([real.shape[0]])
        profit_dnn_2022_case_II = np.zeros([real.shape[0]])
        profit_dnn_2022_case_III = np.zeros([real.shape[0]])
        cost_dnn_2022_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_dnn_2022_case_I = np.zeros([real.shape[0]])
        perfect_foresight_dnn_2022_case_II = np.zeros([real.shape[0]])
        perfect_foresight_dnn_2022_case_III = np.zeros([real.shape[0]])
        perfect_foresight_dnn_2022_case_IV = np.zeros([real.shape[0]])

        profit_xgb_2022_case_I = np.zeros([real.shape[0]])
        profit_xgb_2022_case_II = np.zeros([real.shape[0]])
        profit_xgb_2022_case_III = np.zeros([real.shape[0]])
        cost_xgb_2022_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_xgb_2022_case_I = np.zeros([real.shape[0]])
        perfect_foresight_xgb_2022_case_II = np.zeros([real.shape[0]])
        perfect_foresight_xgb_2022_case_III = np.zeros([real.shape[0]])
        perfect_foresight_xgb_2022_case_IV = np.zeros([real.shape[0]])

        profit_tft_2022_case_I = np.zeros([real.shape[0]])
        profit_tft_2022_case_II = np.zeros([real.shape[0]])
        profit_tft_2022_case_III = np.zeros([real.shape[0]])
        cost_tft_2022_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_tft_2022_case_I = np.zeros([real.shape[0]])
        perfect_foresight_tft_2022_case_II = np.zeros([real.shape[0]])
        perfect_foresight_tft_2022_case_III = np.zeros([real.shape[0]])
        perfect_foresight_tft_2022_case_IV = np.zeros([real.shape[0]])

        profit_cnp_nr_2022_case_I = np.zeros([real.shape[0]])
        profit_cnp_nr_2022_case_II = np.zeros([real.shape[0]])
        profit_cnp_nr_2022_case_III = np.zeros([real.shape[0]])
        cost_cnp_nr_2022_case_IV = np.zeros([real.shape[0]])
        perfect_foresight_cnp_nr_2022_case_I = np.zeros([real.shape[0]])
        perfect_foresight_cnp_nr_2022_case_II = np.zeros([real.shape[0]])
        perfect_foresight_cnp_nr_2022_case_III = np.zeros([real.shape[0]])
        perfect_foresight_cnp_nr_2022_case_IV = np.zeros([real.shape[0]])


        for days_i in range(real_2021.shape[0]):

            realized_prices = real_2021[days_i]
            realized_load = load_2021[days_i]
            realized_renewable_availability = solar_2021[days_i]


            #### CNP

            profit_cnp_2021_case_I[days_i], _, _, _ = solve_case_i(pred_cnp_2021[days_i], realized_prices , is_perfect_foresight=False)
            perfect_foresight_cnp_2021_case_I[days_i], _, _, _ = solve_case_i(real_2021[days_i], realized_prices, is_perfect_foresight=True)

            profit_cnp_2021_case_II[days_i], _, _, _ = solve_case_ii(pred_cnp_2021[days_i], realized_prices, np.sqrt(var_cnp_2021[days_i]), is_perfect_foresight=False)
            perfect_foresight_cnp_2021_case_II[days_i], _, _, _ = solve_case_ii(real_2021[days_i], realized_prices, np.sqrt(var_cnp_2021[days_i]), is_perfect_foresight=True)

            profit_cnp_2021_case_III[days_i], _, _, _ = solve_case_iii(pred_cnp_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=False)
            perfect_foresight_cnp_2021_case_III[days_i], _, _, _ = solve_case_iii(real_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=True)

            cost_cnp_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(pred_cnp_2021[days_i], realized_prices, np.sqrt(var_cnp_2021[days_i]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_cnp_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(real_2021[days_i], realized_prices, np.sqrt(var_cnp_2021[days_i]), fixed_load, is_perfect_foresight=True)

            #### LEAR

            profit_lear_2021_case_I[days_i], _, _, _ = solve_case_i(pred_lear_2021[days_i], realized_prices , is_perfect_foresight=False)
            perfect_foresight_lear_2021_case_I[days_i], _, _, _ = solve_case_i(real_2021[days_i], realized_prices, is_perfect_foresight=True)

            profit_lear_2021_case_II[days_i], _, _, _ = solve_case_ii(pred_lear_2021[days_i], realized_prices, np.sqrt(var_lear_2021[days_i]), is_perfect_foresight=False)
            perfect_foresight_lear_2021_case_II[days_i], _, _, _ = solve_case_ii(real_2021[days_i], realized_prices, np.sqrt(var_lear_2021[days_i]), is_perfect_foresight=True)

            profit_lear_2021_case_III[days_i], _, _, _ = solve_case_iii(pred_lear_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_lear_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=False)
            perfect_foresight_lear_2021_case_III[days_i], _, _, _ = solve_case_iii(real_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_lear_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=True)

            cost_lear_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(pred_lear_2021[days_i], realized_prices, np.sqrt(var_lear_2021[days_i]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_lear_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(real_2021[days_i], realized_prices, np.sqrt(var_lear_2021[days_i]), fixed_load, is_perfect_foresight=True)

            #### XGB

            profit_xgb_2021_case_I[days_i], _, _, _ = solve_case_i(pred_xgb_2021[days_i], realized_prices , is_perfect_foresight=False)
            perfect_foresight_xgb_2021_case_I[days_i], _, _, _ = solve_case_i(real_2021[days_i], realized_prices, is_perfect_foresight=True)

            profit_xgb_2021_case_II[days_i], _, _, _ = solve_case_ii(pred_xgb_2021[days_i], realized_prices, np.sqrt(var_xgb_2021[days_i]), is_perfect_foresight=False)
            perfect_foresight_xgb_2021_case_II[days_i], _, _, _ = solve_case_ii(real_2021[days_i], realized_prices, np.sqrt(var_xgb_2021[days_i]), is_perfect_foresight=True)

            profit_xgb_2021_case_III[days_i], _, _, _ = solve_case_iii(pred_xgb_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_xgb_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=False)
            perfect_foresight_xgb_2021_case_III[days_i], _, _, _ = solve_case_iii(real_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_xgb_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=True)

            cost_xgb_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(pred_xgb_2021[days_i], realized_prices, np.sqrt(var_xgb_2021[days_i]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_xgb_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(real_2021[days_i], realized_prices, np.sqrt(var_xgb_2021[days_i]), fixed_load, is_perfect_foresight=True)

            #### TFT
            
            profit_tft_2021_case_I[days_i], _, _, _ = solve_case_i(pred_tft_2021[days_i], realized_prices , is_perfect_foresight=False)
            perfect_foresight_tft_2021_case_I[days_i], _, _, _ = solve_case_i(real_2021[days_i], realized_prices, is_perfect_foresight=True)

            profit_tft_2021_case_II[days_i], _, _, _ = solve_case_ii(pred_tft_2021[days_i], realized_prices, np.sqrt(var_tft_2021[days_i]), is_perfect_foresight=False)
            perfect_foresight_tft_2021_case_II[days_i], _, _, _ = solve_case_ii(real_2021[days_i], realized_prices, np.sqrt(var_tft_2021[days_i]), is_perfect_foresight=True)

            profit_tft_2021_case_III[days_i], _, _, _ = solve_case_iii(pred_tft_2021[days_i], realized_prices , realized_load, realized_renewable_availability, np.sqrt(var_tft_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=False)
            perfect_foresight_tft_2021_case_III[days_i], _, _, _ = solve_case_iii(real_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_tft_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=True)

            cost_tft_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(pred_tft_2021[days_i], realized_prices, np.sqrt(var_tft_2021[days_i]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_tft_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(real_2021[days_i], realized_prices, np.sqrt(var_tft_2021[days_i]), fixed_load, is_perfect_foresight=True)

            #### blstm
            
            profit_blstm_2021_case_I[days_i], _, _, _ = solve_case_i(pred_blstm_2021[days_i], realized_prices, is_perfect_foresight=False)
            perfect_foresight_blstm_2021_case_I[days_i], _, _, _ = solve_case_i(real_2021[days_i], realized_prices, is_perfect_foresight=True)

            profit_blstm_2021_case_II[days_i], _, _, _ = solve_case_ii(pred_blstm_2021[days_i], realized_prices, np.sqrt(var_blstm_2021[days_i]), is_perfect_foresight=False)
            perfect_foresight_blstm_2021_case_II[days_i], _, _, _ = solve_case_ii(real_2021[days_i], realized_prices, np.sqrt(var_blstm_2021[days_i]), is_perfect_foresight=True)

            profit_blstm_2021_case_III[days_i], _, _, _ = solve_case_iii(pred_blstm_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_blstm_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=False)
            perfect_foresight_blstm_2021_case_III[days_i], _, _, _ = solve_case_iii(real_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_blstm_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=True)

            cost_blstm_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(pred_blstm_2021[days_i], realized_prices, np.sqrt(var_blstm_2021[days_i]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_blstm_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(real_2021[days_i], realized_prices, np.sqrt(var_blstm_2021[days_i]), fixed_load, is_perfect_foresight=True)

            #### CNP_NO_Regimes

            profit_cnp_nr_2021_case_I[days_i], _, _, _ = solve_case_i(pred_cnp_nr_2021[days_i], realized_prices, is_perfect_foresight=False)
            perfect_foresight_cnp_nr_2021_case_I[days_i], _, _, _ = solve_case_i(real_2021[days_i], realized_prices, is_perfect_foresight=True)

            profit_cnp_nr_2021_case_II[days_i], _, _, _ = solve_case_ii(pred_cnp_nr_2021[days_i], realized_prices, np.sqrt(var_cnp_nr_2021[days_i]), is_perfect_foresight=False)
            perfect_foresight_cnp_nr_2021_case_II[days_i], _, _, _ = solve_case_ii(real_2021[days_i], realized_prices, np.sqrt(var_cnp_nr_2021[days_i]), is_perfect_foresight=True)

            profit_cnp_nr_2021_case_III[days_i], _, _, _ = solve_case_iii(pred_cnp_nr_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_nr_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=False)
            perfect_foresight_cnp_nr_2021_case_III[days_i], _, _, _ = solve_case_iii(real_2021[days_i], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_nr_2021[days_i]), load_2021[days_i], solar_2021[days_i], is_perfect_foresight=True)

            cost_cnp_nr_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(pred_cnp_nr_2021[days_i], realized_prices, np.sqrt(var_cnp_nr_2021[days_i]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_cnp_nr_2021_case_IV[days_i], _, _, _, _ = solve_case_iv(real_2021[days_i], realized_prices, np.sqrt(var_cnp_nr_2021[days_i]), fixed_load, is_perfect_foresight=True)

        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'profit_cnp_2021_case_I.txt', profit_cnp_2021_case_I)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'profit_cnp_2021_case_II.txt', profit_cnp_2021_case_II)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'profit_cnp_2021_case_III.txt', profit_cnp_2021_case_III)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'cost_cnp_2021_case_IV.txt', cost_cnp_2021_case_IV)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_2021_case_I.txt', perfect_foresight_cnp_2021_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_2021_case_II.txt', perfect_foresight_cnp_2021_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_2021_case_III.txt', perfect_foresight_cnp_2021_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_2021_case_IV.txt', perfect_foresight_cnp_2021_case_IV)

        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'profit_tft_2021_case_I.txt', profit_tft_2021_case_I)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'profit_tft_2021_case_II.txt', profit_tft_2021_case_II)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'profit_tft_2021_case_III.txt', profit_tft_2021_case_III)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'cost_tft_2021_case_IV.txt', cost_tft_2021_case_IV)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_2021_case_I.txt', perfect_foresight_tft_2021_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_2021_case_II.txt', perfect_foresight_tft_2021_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_2021_case_III.txt', perfect_foresight_tft_2021_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_2021_case_IV.txt', perfect_foresight_tft_2021_case_IV)
        


        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'profit_lear_2021_case_I.txt', profit_lear_2021_case_I)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'profit_lear_2021_case_II.txt', profit_lear_2021_case_II)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'profit_lear_2021_case_III.txt', profit_lear_2021_case_III)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'cost_lear_2021_case_IV.txt', cost_lear_2021_case_IV)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_2021_case_I.txt', perfect_foresight_lear_2021_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_2021_case_II.txt', perfect_foresight_lear_2021_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_2021_case_III.txt', perfect_foresight_lear_2021_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_2021_case_IV.txt', perfect_foresight_lear_2021_case_IV)


        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'profit_blstm_2021_case_I.txt', profit_blstm_2021_case_I)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'profit_blstm_2021_case_II.txt', profit_blstm_2021_case_II)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'profit_blstm_2021_case_III.txt', profit_blstm_2021_case_III)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'cost_blstm_2021_case_IV.txt', cost_blstm_2021_case_IV)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_2021_case_I.txt', perfect_foresight_blstm_2021_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_2021_case_II.txt', perfect_foresight_blstm_2021_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_2021_case_III.txt', perfect_foresight_blstm_2021_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_2021_case_IV.txt', perfect_foresight_blstm_2021_case_IV)


        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'profit_xgb_2021_case_I.txt', profit_xgb_2021_case_I)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'profit_xgb_2021_case_II.txt', profit_xgb_2021_case_II)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'profit_xgb_2021_case_III.txt', profit_xgb_2021_case_III)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'cost_xgb_2021_case_IV.txt', cost_xgb_2021_case_IV)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_2021_case_I.txt', perfect_foresight_xgb_2021_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_2021_case_II.txt', perfect_foresight_xgb_2021_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_2021_case_III.txt', perfect_foresight_xgb_2021_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_2021_case_IV.txt', perfect_foresight_xgb_2021_case_IV)


        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'profit_cnp_no_regime_2021_case_I.txt', profit_cnp_nr_2021_case_I)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'profit_cnp_no_regime_2021_case_II.txt', profit_cnp_nr_2021_case_II)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'profit_cnp_no_regime_2021_case_III.txt', profit_cnp_nr_2021_case_III)
        #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'cost_cnp_no_regime_2021_case_IV.txt', cost_cnp_nr_2021_case_IV)


        real_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/real_data/real_germany_2021.txt')[3:365,:24]
        real_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/real_data/real_germany_2022.txt')[3:365,:24]
        load_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_load_data_'+cnt+'_2021.txt')[3:365,:24]
        solar_2021 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_pv_data_'+cnt+'_2021.txt')[3:365,:24]
        load_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_load_data_'+cnt+'_2022.txt')[3:365,:24]
        solar_2022 = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/sim_pv_data_'+cnt+'_2022.txt')[3:365,:24]
    
        for days_d in range(pred_dnn_2021.shape[0]):
            realized_prices = real_2021[days_d]
            realized_load = load_2021[days_d]
            realized_renewable_availability = solar_2021[days_d]
            #### DNN

            profit_dnn_2021_case_I[days_d], _, _, _ = solve_case_i(pred_dnn_2021[days_d], realized_prices , is_perfect_foresight=False)
            perfect_foresight_dnn_2021_case_I[days_d], _, _, _ = solve_case_i(real_2021[days_d], realized_prices, is_perfect_foresight=True)

            profit_dnn_2021_case_II[days_d], _, _, _ = solve_case_ii(pred_dnn_2021[days_d], realized_prices, np.sqrt(var_dnn_2021[days_d]), is_perfect_foresight=False)
            perfect_foresight_dnn_2021_case_II[days_d], _, _, _ = solve_case_ii(real_2021[days_d], realized_prices, np.sqrt(var_dnn_2021[days_d]), is_perfect_foresight=True)

            profit_dnn_2021_case_III[days_d], _, _, _ = solve_case_iii(pred_dnn_2021[days_d], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_dnn_2021[days_d]), load_2021[days_d], solar_2021[days_d], is_perfect_foresight=False)
            perfect_foresight_dnn_2021_case_III[days_d], _, _, _ = solve_case_iii(real_2021[days_d], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_dnn_2021[days_d]), load_2021[days_d], solar_2021[days_d], is_perfect_foresight=True)

            cost_dnn_2021_case_IV[days_d], _, _, _, _ = solve_case_iv(pred_dnn_2021[days_d], realized_prices, np.sqrt(var_dnn_2021[days_d]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_dnn_2021_case_IV[days_d], _, _, _, _ = solve_case_iv(real_2021[days_d], realized_prices, np.sqrt(var_dnn_2021[days_d]), fixed_load, is_perfect_foresight=True)
      np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'profit_dnn_2021_case_I.txt', profit_dnn_2021_case_I)
      np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'profit_dnn_2021_case_II.txt', profit_dnn_2021_case_II)
      np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'profit_dnn_2021_case_III.txt', profit_dnn_2021_case_III)
      np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'cost_dnn_2021_case_IV.txt', cost_dnn_2021_case_IV)
      np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_2021_case_I.txt', perfect_foresight_dnn_2021_case_I)
      np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_2021_case_II.txt', perfect_foresight_dnn_2021_case_II)
      np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_2021_case_III.txt', perfect_foresight_dnn_2021_case_III)
      np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_2021_case_IV.txt', perfect_foresight_dnn_2021_case_IV)




        for days_ii in range(real_2022.shape[0]):

            realized_prices = real_2022[days_ii]
            realized_load = load_2022[days_ii]
            realized_renewable_availability = solar_2022[days_ii]


            #### CNP

            profit_cnp_2022_case_I[days_ii], _, _, _ = solve_case_i(pred_cnp_2022[days_ii], realized_prices , is_perfect_foresight=False)
            perfect_foresight_cnp_2022_case_I[days_ii], _, _, _ = solve_case_i(real_2022[days_ii], realized_prices, is_perfect_foresight=True)

            profit_cnp_2022_case_II[days_ii], _, _, _ = solve_case_ii(pred_cnp_2022[days_ii], realized_prices, np.sqrt(var_cnp_2022[days_ii]), is_perfect_foresight=False)
            perfect_foresight_cnp_2022_case_II[days_ii], _, _, _ = solve_case_ii(real_2022[days_ii], realized_prices, np.sqrt(var_cnp_2022[days_ii]), is_perfect_foresight=True)

            profit_cnp_2022_case_III[days_ii], _, _, _ = solve_case_iii(pred_cnp_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=False)
            perfect_foresight_cnp_2022_case_III[days_ii], _, _, _ = solve_case_iii(real_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=True)

            cost_cnp_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(pred_cnp_2022[days_ii], realized_prices, np.sqrt(var_cnp_2022[days_ii]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_cnp_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(real_2022[days_ii], realized_prices, np.sqrt(var_cnp_2022[days_ii]), fixed_load, is_perfect_foresight=True)

            #### LEAR

            profit_lear_2022_case_I[days_ii], _, _, _ = solve_case_i(pred_lear_2022[days_ii], realized_prices , is_perfect_foresight=False)
            perfect_foresight_lear_2022_case_I[days_ii], _, _, _ = solve_case_i(real_2022[days_ii], realized_prices, is_perfect_foresight=True)

            profit_lear_2022_case_II[days_ii], _, _, _ = solve_case_ii(pred_lear_2022[days_ii], realized_prices, np.sqrt(var_lear_2022[days_ii]), is_perfect_foresight=False)
            perfect_foresight_lear_2022_case_II[days_ii], _, _, _ = solve_case_ii(real_2022[days_ii], realized_prices, np.sqrt(var_lear_2022[days_ii]), is_perfect_foresight=True)

            profit_lear_2022_case_III[days_ii], _, _, _ = solve_case_iii(pred_lear_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_lear_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=False)
            perfect_foresight_lear_2022_case_III[days_ii], _, _, _ = solve_case_iii(real_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_lear_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=True)

            cost_lear_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(pred_lear_2022[days_ii], realized_prices, np.sqrt(var_lear_2022[days_ii]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_lear_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(real_2022[days_ii], realized_prices, np.sqrt(var_lear_2022[days_ii]), fixed_load, is_perfect_foresight=True)

            #### DNN

            profit_dnn_2022_case_I[days_ii], _, _, _ = solve_case_i(pred_dnn_2022[days_ii], realized_prices , is_perfect_foresight=False)
            perfect_foresight_dnn_2022_case_I[days_ii], _, _, _ = solve_case_i(real_2022[days_ii], realized_prices, is_perfect_foresight=True)

            profit_dnn_2022_case_II[days_ii], _, _, _ = solve_case_ii(pred_dnn_2022[days_ii], realized_prices, np.sqrt(var_dnn_2022[days_ii]), is_perfect_foresight=False)
            perfect_foresight_dnn_2022_case_II[days_ii], _, _, _ = solve_case_ii(real_2022[days_ii], realized_prices, np.sqrt(var_dnn_2022[days_ii]), is_perfect_foresight=True)

            profit_dnn_2022_case_III[days_ii], _, _, _ = solve_case_iii(pred_dnn_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_dnn_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=False)
            perfect_foresight_dnn_2022_case_III[days_ii], _, _, _ = solve_case_iii(real_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_dnn_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=True)

            cost_dnn_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(pred_dnn_2022[days_ii], realized_prices, np.sqrt(var_dnn_2022[days_ii]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_dnn_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(real_2022[days_ii], realized_prices, np.sqrt(var_dnn_2022[days_ii]), fixed_load, is_perfect_foresight=True)

            #### XGB

            profit_xgb_2022_case_I[days_ii], _, _, _ = solve_case_i(pred_xgb_2022[days_ii], realized_prices , is_perfect_foresight=False)
            perfect_foresight_xgb_2022_case_I[days_ii], _, _, _ = solve_case_i(real_2022[days_ii], realized_prices, is_perfect_foresight=True)

            profit_xgb_2022_case_II[days_ii], _, _, _ = solve_case_ii(pred_xgb_2022[days_ii], realized_prices, np.sqrt(var_xgb_2022[days_ii]), is_perfect_foresight=False)
            perfect_foresight_xgb_2022_case_II[days_ii], _, _, _ = solve_case_ii(real_2022[days_ii], realized_prices, np.sqrt(var_xgb_2022[days_ii]), is_perfect_foresight=True)

            profit_xgb_2022_case_III[days_ii], _, _, _ = solve_case_iii(pred_xgb_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_xgb_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=False)
            perfect_foresight_xgb_2022_case_III[days_ii], _, _, _ = solve_case_iii(real_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_xgb_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=True)

            cost_xgb_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(pred_xgb_2022[days_ii], realized_prices, np.sqrt(var_xgb_2022[days_ii]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_xgb_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(real_2022[days_ii], realized_prices, np.sqrt(var_xgb_2022[days_ii]), fixed_load, is_perfect_foresight=True)

            #### TFT
            
            profit_tft_2022_case_I[days_ii], _, _, _ = solve_case_i(pred_tft_2022[days_ii], realized_prices , is_perfect_foresight=False)
            perfect_foresight_tft_2022_case_I[days_ii], _, _, _ = solve_case_i(real_2022[days_ii], realized_prices, is_perfect_foresight=True)

            profit_tft_2022_case_II[days_ii], _, _, _ = solve_case_ii(pred_tft_2022[days_ii], realized_prices, np.sqrt(var_tft_2022[days_ii]), is_perfect_foresight=False)
            perfect_foresight_tft_2022_case_II[days_ii], _, _, _ = solve_case_ii(real_2022[days_ii], realized_prices, np.sqrt(var_tft_2022[days_ii]), is_perfect_foresight=True)

            profit_tft_2022_case_III[days_ii], _, _, _ = solve_case_iii(pred_tft_2022[days_ii], realized_prices , realized_load, realized_renewable_availability, np.sqrt(var_tft_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=False)
            perfect_foresight_tft_2022_case_III[days_ii], _, _, _ = solve_case_iii(real_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_tft_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=True)

            cost_tft_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(pred_tft_2022[days_ii], realized_prices, np.sqrt(var_tft_2022[days_ii]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_tft_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(real_2022[days_ii], realized_prices, np.sqrt(var_tft_2022[days_ii]), fixed_load, is_perfect_foresight=True)

            #### blstm
            
            profit_blstm_2022_case_I[days_ii], _, _, _ = solve_case_i(pred_blstm_2022[days_ii], realized_prices, is_perfect_foresight=False)
            perfect_foresight_blstm_2022_case_I[days_ii], _, _, _ = solve_case_i(real_2022[days_ii], realized_prices, is_perfect_foresight=True)

            profit_blstm_2022_case_II[days_ii], _, _, _ = solve_case_ii(pred_blstm_2022[days_ii], realized_prices, np.sqrt(var_blstm_2022[days_ii]), is_perfect_foresight=False)
            perfect_foresight_blstm_2022_case_II[days_ii], _, _, _ = solve_case_ii(real_2022[days_ii], realized_prices, np.sqrt(var_blstm_2022[days_ii]), is_perfect_foresight=True)

            profit_blstm_2022_case_III[days_ii], _, _, _ = solve_case_iii(pred_blstm_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_blstm_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=False)
            perfect_foresight_blstm_2022_case_III[days_ii], _, _, _ = solve_case_iii(real_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_blstm_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=True)

            cost_blstm_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(pred_blstm_2022[days_ii], realized_prices, np.sqrt(var_blstm_2022[days_ii]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_blstm_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(real_2022[days_ii], realized_prices, np.sqrt(var_blstm_2022[days_ii]), fixed_load, is_perfect_foresight=True)

            #### CNP_NO_Regimes

            profit_cnp_nr_2022_case_I[days_ii], _, _, _ = solve_case_i(pred_cnp_nr_2022[days_ii], realized_prices, is_perfect_foresight=False)
            perfect_foresight_cnp_nr_2022_case_I[days_ii], _, _, _ = solve_case_i(real_2022[days_ii], realized_prices, is_perfect_foresight=True)

            profit_cnp_nr_2022_case_II[days_ii], _, _, _ = solve_case_ii(pred_cnp_nr_2022[days_ii], realized_prices, np.sqrt(var_cnp_nr_2022[days_ii]), is_perfect_foresight=False)
            perfect_foresight_cnp_nr_2022_case_II[days_ii], _, _, _ = solve_case_ii(real_2022[days_ii], realized_prices, np.sqrt(var_cnp_nr_2022[days_ii]), is_perfect_foresight=True)

            profit_cnp_nr_2022_case_III[days_ii], _, _, _ = solve_case_iii(pred_cnp_nr_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_nr_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=False)
            perfect_foresight_cnp_nr_2022_case_III[days_ii], _, _, _ = solve_case_iii(real_2022[days_ii], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var_cnp_nr_2022[days_ii]), load_2022[days_ii], solar_2022[days_ii], is_perfect_foresight=True)

            cost_cnp_nr_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(pred_cnp_nr_2022[days_ii], realized_prices, np.sqrt(var_cnp_nr_2022[days_ii]), fixed_load, is_perfect_foresight=False)
            perfect_foresight_cnp_nr_2022_case_IV[days_ii], _, _, _, _ = solve_case_iv(real_2022[days_ii], realized_prices, np.sqrt(var_cnp_nr_2022[days_ii]), fixed_load, is_perfect_foresight=True)
        """
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'profit_cnp_2022_case_I.txt', profit_cnp_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'profit_cnp_2022_case_II.txt', profit_cnp_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'profit_cnp_2022_case_III.txt', profit_cnp_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'cost_cnp_2022_case_IV.txt', cost_cnp_2022_case_IV)

        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'profit_tft_2022_case_I.txt', profit_tft_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'profit_tft_2022_case_II.txt', profit_tft_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'profit_tft_2022_case_III.txt', profit_tft_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'cost_tft_2022_case_IV.txt', cost_tft_2022_case_IV)

        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'profit_dnn_2022_case_I.txt', profit_dnn_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'profit_dnn_2022_case_II.txt', profit_dnn_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'profit_dnn_2022_case_III.txt', profit_dnn_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'cost_dnn_2022_case_IV.txt', cost_dnn_2022_case_IV)

        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'profit_lear_2022_case_I.txt', profit_lear_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'profit_lear_2022_case_II.txt', profit_lear_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'profit_lear_2022_case_III.txt', profit_lear_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'cost_lear_2022_case_IV.txt', cost_lear_2022_case_IV)

        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'profit_blstm_2022_case_I.txt', profit_blstm_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'profit_blstm_2022_case_II.txt', profit_blstm_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'profit_blstm_2022_case_III.txt', profit_blstm_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'cost_blstm_2022_case_IV.txt', cost_blstm_2022_case_IV)

        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'profit_xgb_2022_case_I.txt', profit_xgb_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'profit_xgb_2022_case_II.txt', profit_xgb_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'profit_xgb_2022_case_III.txt', profit_xgb_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'cost_xgb_2022_case_IV.txt', cost_xgb_2022_case_IV)

        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'profit_cnp_no_regime_2022_case_I.txt', profit_cnp_nr_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'profit_cnp_no_regime_2022_case_II.txt', profit_cnp_nr_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'profit_cnp_no_regime_2022_case_III.txt', profit_cnp_nr_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_no_regime_trading/'+'cost_cnp_no_regime_2022_case_IV.txt', cost_cnp_nr_2022_case_IV)
        """
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_2022_case_I.txt', perfect_foresight_dnn_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_2022_case_II.txt', perfect_foresight_dnn_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_2022_case_III.txt', perfect_foresight_dnn_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/dnn_trading/'+'pf_dnn_2022_case_IV.txt', perfect_foresight_dnn_2022_case_IV)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_2022_case_I.txt', perfect_foresight_tft_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_2022_case_II.txt', perfect_foresight_tft_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_2022_case_III.txt', perfect_foresight_tft_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/tft_trading/'+'pf_tft_2022_case_IV.txt', perfect_foresight_tft_2022_case_IV)

        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_2022_case_I.txt', perfect_foresight_cnp_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_2022_case_II.txt', perfect_foresight_cnp_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_2022_case_III.txt', perfect_foresight_cnp_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/cnp_trading/'+'pf_cnp_2022_case_IV.txt', perfect_foresight_cnp_2022_case_IV)
        
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_2022_case_I.txt', perfect_foresight_lear_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_2022_case_II.txt', perfect_foresight_lear_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_2022_case_III.txt', perfect_foresight_lear_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/lear_trading/'+'pf_lear_2022_case_IV.txt', perfect_foresight_lear_2022_case_IV)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_2022_case_I.txt', perfect_foresight_blstm_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_2022_case_II.txt', perfect_foresight_blstm_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_2022_case_III.txt', perfect_foresight_blstm_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/blstm_trading/'+'pf_blstm_2022_case_IV.txt', perfect_foresight_blstm_2022_case_IV)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_2022_case_I.txt', perfect_foresight_xgb_2022_case_I)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_2022_case_II.txt', perfect_foresight_xgb_2022_case_II)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_2022_case_III.txt', perfect_foresight_xgb_2022_case_III)
        np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_'+cnt+'/xgb_trading/'+'pf_xgb_2022_case_IV.txt', perfect_foresight_xgb_2022_case_IV)

