# A-recent-global-increase-in-vegetation-resilience
Data and code for "A recent global increase in vegetation resilience "

#Code
AR_test.py - Estimates vegetation resilience from monthly kernel normalized difference vegetation index (kNDVI) data—detrended and deseasonalized—using a sliding-window approach based on lagged first-order autocorrelation.
XGB_SHAP_test.py - Determines the relative importance of key factors driving changes in vegetation resilience, using XGBoost and SHAP.


#Data
AR_Xyearsmean_datatype20002023.tif - Mean AR(1) values for 2000–2023, calculated from kNDVI, SIF, and GPP data using sliding windows of 3, 5, and 7 years.
AR_SSPXXX_mean_20242100.tif - Mean AR(1) values for 2024–2100, derived from the mean GPP of multiple models under different SSP scenarios (SSP126, SSP245, SSP370, SSP585) using a 5-year sliding window.

