Replication Package for "Regime-aware conditional neural processes with multi-criteria decision support for operational electricity price forecasting". Clone the whole repo so that all required files and folders remain in the same directory. Run replication\_package.py file to replicate the results in the paper.

 

 **Note on Case II, Case III, and TOPSIS**



In the section 6, "Operational cases", Case II and Case III are distinct. Case II uses uncertainty-aware arbitrage, whereas Case III additionally includes load and solar grid-support terms. Case III charging adjustment does not change the optimal battery dispatch, so Case II and Case III can produce identical or nearly identical values. This is expected from the linear-programming formulation and does not indicate an implementation error.

The TOPSIS criteria should be interpreted as a multi-criteria summary, not as a set of fully independent signals. MAE, RMSE, and SMAPE are related forecasting-error measures, although they emphasize different aspects of error, and Case II/III values coincide whenever the grid-support terms do not alter the optimal dispatch. The ranking still aggregates forecasting and operational performance under the stated criterion design, and the reported robustness checks reduce dependence on any single weighting choice

