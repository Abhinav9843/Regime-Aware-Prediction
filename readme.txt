Python Code for Manuscript: Regime-Aware Conditional Neural Processes with Multi-Criteria Decision Support for 
			    Operational Electricity Price Forecasting

In this study there are several models are compared. All of the codes are given in this repo. A small demo for operational studies are also given in "Operational_cases_demo" folder.



This main folder has a subfolder named: "Codes" which has all of the models coded in a python script using the following packages below. We request the user to install these packages beforehand. The data used in this study are the German, French and Norwegian electricity data which in the subfolder named: "Dataset". The German data is publicly available from the Bundesnetzagetur (English translation: German Federal Agency) through the webpage German Data whereas the French and Norwegian Data are available from ENTSOE Transparency platform through the webpage: ENSTOE Transparency Platform .
 

Third-party packages detected from the provided imports:

cvxpy
datetime
epftoolbox (follow https://epftoolbox.readthedocs.io/en/latest/modules/data.html)
lightning
matplotlib
numpy
pandas
pytictoc
pytorch_forecasting
scikit_posthocs
scipy
seaborn
sklearn
statsmodels
torch
TicToc




========================================================
Datasets
=========================================================
For Germany the a day ahead electricity price data, residual load data forecast data and total renewable production (solar + wind) from 2017 to 2023 is required. The data can be downloaded from https://www.smard.de/en/downloadcenter/download-market-data/
For France the a day ahead electricity price data, load forecast and generation forecast data from 2017 to 2023 is required. The data can be downloaded from https://transparency.entsoe.eu/
For Norway we used Bidding Zone NO2 for the a day ahead electricity price data, load forecast and generation forecast data from 2017 to 2023 is required. The data can be downloaded from https://transparency.entsoe.eu/.

The date (in format dd/mm/yyyy or dd-mm-yyyy) must be the first column of the dataset.


