import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = "us.csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# Convert date column to datetime and set it as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Plot the cases to visualize trends and seasonality
plt.figure(figsize=(12, 6))
plt.plot(df['cases'], label="Cases")
plt.title("COVID-19 Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.legend()
plt.show()

# First seasonal decomposition with 180-day period to remove long-term seasonality
decomp_180 = seasonal_decompose(df['cases'], period=180, model='additive')

# Remove 180-day seasonality
residual_180 = decomp_180.resid.dropna()

# Second seasonal decomposition with 7-day period to remove short-term seasonality
decomp_7 = seasonal_decompose(residual_180, period=7, model='additive')

# Function to perform stationarity tests
def check_stationarity(series):
    adf_test = adfuller(series.dropna())
    kpss_test = kpss(series.dropna(), regression='c', nlags='auto')

    results = {
        "ADF Test (H0: Non-stationary)": {"Test Statistic": adf_test[0], "p-value": adf_test[1]},
        "KPSS Test (H0: Stationary)": {"Test Statistic": kpss_test[0], "p-value": kpss_test[1]}
    }
    
    return results

# Check stationarity of final residuals (after both decompositions)
stationarity_results = check_stationarity(decomp_7.resid)
print(stationarity_results)

# Fit an ARMA model (since the data is stationary, d=0)
arma_order = (7, 0, 7)  # Using 7 to capture weekly seasonality effects
arma_model = ARIMA(decomp_7.resid.dropna(), order=arma_order).fit()

# Get fitted values
fitted_values = arma_model.fittedvalues

# Align index of fitted values with residuals
residuals_aligned = decomp_7.resid.dropna()
fitted_values = fitted_values.loc[residuals_aligned.index]

# Plot actual residuals vs fitted ARMA model
plt.figure(figsize=(12, 6))
plt.plot(residuals_aligned.index, residuals_aligned, label="Actual Residuals", alpha=0.6)
plt.plot(fitted_values.index, fitted_values, label="Fitted ARMA Model", linestyle="dashed")
plt.title("Actual vs Fitted ARMA Model")
plt.xlabel("Date")
plt.ylabel("Cases Residuals")
plt.legend()
plt.show()

# Reconstruct the predicted cases correctly by adding back trend and seasonal components
corrected_predicted_cases = (decomp_180.trend + decomp_7.seasonal + fitted_values).dropna()

# Align index of predicted cases with actual cases
actual_cases_aligned = df['cases'].loc[corrected_predicted_cases.index]
 
# Plot actual cases vs corrected predicted cases
plt.figure(figsize=(12, 6))
plt.plot(actual_cases_aligned.index, actual_cases_aligned, label="Actual Cases", alpha=0.6)
plt.plot(corrected_predicted_cases.index, corrected_predicted_cases, label="Corrected Predicted Cases (Decomposition + ARMA)", linestyle="dashed")
plt.title("Actual Cases vs Corrected Predicted Cases")
plt.xlabel("Date")
plt.ylabel("Cases")
plt.legend()
plt.show()

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((actual_cases_aligned - corrected_predicted_cases) / actual_cases_aligned)) * 100
print(f"MAPE: {mape:.2f}%")
