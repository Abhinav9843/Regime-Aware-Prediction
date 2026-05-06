Please follow the "readme.txt" file for Data set download.


Note on Case II, Case III, and TOPSIS
Case II and Case III are distinct Section 6 formulations: Case II uses uncertainty-aware arbitrage, whereas Case III additionally includes load and solar grid-support terms. In some market-year cases, the Case III charging adjustment does not change the optimal battery dispatch, so Case II and Case III can produce identical or nearly identical values. This is expected from the linear-programming formulation and does not indicate an implementation error.


The TOPSIS criteria should be interpreted as a multi-criteria summary, not as a set of fully independent signals. MAE, RMSE, and SMAPE are related forecasting-error measures, although they emphasize different aspects of error, and Case II/III values coincide whenever the grid-support terms do not alter the optimal dispatch. The ranking still aggregates forecasting and operational performance under the stated criterion design, and the reported robustness checks reduce dependence on any single weighting choice.
