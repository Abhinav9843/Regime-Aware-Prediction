from trading_strategies_I_IV import*

sensetivity_cases = ['case-I','case-II','case-III','case-IV','case-V','case-VI','case-VII','case-VIII','case-IX','case-X','case-XI']
real = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/real_data/real_sensetivity.txt')[:179,:24]
load = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/load_sensetivity.txt')[:179,:24]
solar = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Strategies/pv_sensetivity.txt')[:179,:24]

profit_cnp_case_III = np.zeros([real.shape[0]])
perfect_foresight_cnp_case_III = np.zeros([real.shape[0]])
profit_case_III = np.zeros([len(sensetivity_cases),1])
for idx, cs in enumerate(sensetivity_cases):
    pred = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/Result_germany_sensetivity/'+ cs + '_new/pred_cnp_'+cs+'.txt')[:179,:24]
    var = np.loadtxt('C:/Users/abhin/Downloads/R-CNP-V2/Prediction Result/CNP/Result_CNP_HYPER_Tuned/Result_germany_sensetivity/'+ cs + '_new/var_cnp_'+cs+'.txt')[:179,:24]

    for days in range(pred.shape[0]):
        realized_prices = real[days]
        realized_load = load[days]
        realized_renewable_availability = solar[days]
        
        profit_cnp_case_III[days], _, _, _ = solve_case_iii(pred[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var[days]), load[days], solar[days], is_perfect_foresight=False)
        perfect_foresight_cnp_case_III[days], _, _, _ = solve_case_iii(real[days], realized_prices, realized_load, realized_renewable_availability, np.sqrt(var[days]), load[days], solar[days], is_perfect_foresight=True)
    profit_case_III[idx] = np.mean(profit_cnp_case_III)
    print(profit_case_III)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_Sensetivity/profit_'+cs+'.txt', profit_cnp_case_III)
    #np.savetxt('C:/Users/abhin/Downloads/R-CNP-V2/Trading_Result/Trading_Result_Sensetivity/perfect_foresight_'+cs+'.txt', perfect_foresight_cnp_case_III)

