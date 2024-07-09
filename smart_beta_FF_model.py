#%% Preliminaries ---------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yahooFinance
from fredapi import Fred
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Set path to working directory of this script file
path_file = os.path.dirname(os.path.abspath(__file__))


#%% Import Data -----------------------------------------------------------------------------------
#>> Load stock returns (Date, Ticker, and Return columns)
df_QQQ = yahooFinance.Ticker("QQQ").history(start='1990-01-01', interval='1mo', actions=True)
# Compute monthly log returns
df_QQQ["Returns"] = np.log(df_QQQ["Close"]/df_QQQ["Close"].shift(1))
df_QQQ = df_QQQ[["Close", "Volume", "Returns"]]
print(df_QQQ.head())
# Extract the date of the first observation
start_date = datetime.strftime(df_QQQ.index[0], '%m/%d/%Y'); start_date

#>> Import Ken French's data directly
url_FF5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
df_FF5 = pd.read_csv(url_FF5, compression="zip", skiprows=3)
print(df_FF5.head())


#%% Data Cleaning ---------------------------------------------------------------------------------
#>> Clean FF data
df_FF5.rename(columns={'Unnamed: 0':'Date'}, inplace=True)
string_location = df_FF5[df_FF5['Date'].str.contains("Annual Factors: January-December", na=False)].index[0]
df_FF5 = df_FF5[:string_location]
df_FF5['Date'] = pd.to_datetime(df_FF5['Date'], format='%Y%m')
df_FF5.set_index('Date', inplace=True)
df_FF5 = df_FF5.apply(pd.to_numeric, errors='coerce')

# Normalize timezone in datetype indexes
df_FF5 = df_FF5.tz_localize(None)
df_QQQ = df_QQQ.tz_localize(None)

#>> Combine data series & compute excess returns
df = df_FF5.join(df_QQQ, how='inner')
df["excess_return"] = df["Returns"] - df["RF"] / 100
# Drop any rows with missing values
df = df.dropna()


#%% Build Model(s) --------------------------------------------------------------------------------
# Define the independent variables (Fama-French factors)
X = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
# Add a constant (intercept) to the model
X = sm.add_constant(X)
# Define the dependent variable (Excess Return)
y = df['excess_return']

# Perform the regression
model = sm.OLS(y, X).fit()
regression_summary = model.summary()
regression_summary


#%% Conclusion ------------------------------------------------------------------------------------
#>> Does QQQ have alpha?
# Alpha: QQQ has a statistically significant positive alpha, indicating it outperforms the Fama-French 5-factor model
# Factor Loadings: QQQ is strongly influenced by the market factor and negatively by the value factor, aligning with its growth-oriented strategy.

#>> Why QQQ Might Have Alpha
# Growth Exposure: QQQ is heavily weighted towards high-growth technology companies (approx. 61.86%), which may have outperformed the broader market due to innovation & strong earnings growth.
# Market Sentiment: Investor sentiment towards technology & growth stocks might have contributed to QQQ's consistent outperformance.

#>> What do QQQ's factor loadings mean?
# 1. Alpha (Intercept)
# Coefficient (const): 0.0021
# p-value: 0.125
# The alpha is 0.21% per month, which is NOT statistically significant at the 5% level (p < 0.05). This indicates that QQQ has a positive alpha, suggesting it outperforms what the Fama-French 5-factor model predicts.
# The coefficient for SMB is not statistically significant, suggesting that QQQ's returns are not strongly influenced by the size factor.

# 2. Market Factor (Mkt-RF)
# Coefficient: 0.0119
# p-value: 0.000
# The coefficient is highly significant, indicating that QQQ's returns are strongly related to the market risk premium. The positive coefficient implies that QQQ generally moves in line with the overall market.

# 3. Size Factor (SMB)
# Coefficient: -0.0010
# p-value: 0.048
# The coefficient for SMB is negative & statistically significant, indicating that QQQ tends to underperform when the size factor performs well, consistent with QQQ's growth-oriented profile.

# 4. Value Factor (HML)
# Coefficient: -0.0046
# p-value: 0.000
# The coefficient for HML is negative & statistically significant, indicating that QQQ tends to underperform when the value factor performs well.

# 5. Profitability Factor (RMW)
# Coefficient: -0.0044
# p-value: 0.000
# The coefficient for RMW is negative & statistically significant, suggesting that QQQ's returns are strongly influenced by the profitability factor.

# 6. Investment Factor (CMA)
# Coefficient: -0.0024
# p-value: 0.002
# The coefficient for CMA is statistically significant, indicating that QQQ's returns are strongly influenced by the investment factor.

#>> Is 20 years worth of monthly data enough?
# Data Sufficiency: The 20 years of monthly data appear to be sufficient given the high R-squared value (0.896) and the statistical significance of several factors.
#   > also need to consider any structural changes that might've occurred during the full time horizon which we're considering (from 2000-2020)

#%% Notes -----------------------------------------------------------------------------------------
# References:
    # https://mortada.net/python-api-for-fred.html

# Mkt_RF (Rm - Rf) is the return spread between the capitalization-weighted stock market and cash
# SMB is the return spread of small minus large stocks (i.e. the size effect)
# HML is the return spread of cheap minus expensive stocks (i.e. the value effect)
# RMW is the return spread of the most profitable firms minus the least profitable
# CMA is the return spread of firms that invest conservatively minus aggressively (AQR, 2014)
