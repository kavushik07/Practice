import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Fetch NIFTY 50 data
ticker = "AAPL"  # NIFTY 50 index symbol
data = yf.download(ticker, start="2014-01-01", end="2024-01-1", interval='1mo',progress=False)

# Step 2: Calculate daily percentage returns
data['Return'] = ((data['Close'] - data['Open']) / data['Open']) * 100

# Handle missing values
data['Return'] = data['Return'].fillna(method='ffill')
returns = data['Return'].values
n = len(returns)

print(f"Number of data points: {n}")
# Step 3: Calculate ACF manually
def calculate_acf(series, max_lag):
    n = len(series)
    mean = np.mean(series)
    denominator = np.sum((series - mean) ** 2)
    acf = []

    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)  # ACF at lag 0 is always 1
        else:
            numerator = np.sum((series[lag:] - mean) * (series[:-lag] - mean))
            acf.append(numerator / denominator if denominator != 0 else 0)

    return np.array(acf)

# Step 4: Calculate PACF manually using Durbin-Levinson algorithm
def calculate_pacf(acf, max_lag):
    pacf = [1.0]
    phi = [0] * max_lag  # Initialize phi coefficients

    for k in range(1, max_lag + 1):
        if k == 1:
            phi[0] = acf[1]
        else:
            # Calculate phi_kk
            numerator = acf[k] - sum(phi[j-1] * acf[k-j] for j in range(1, k))
            denominator = 1 - sum(phi[j-1] * acf[j] for j in range(1, k))
            phi_kk = numerator / denominator if denominator != 0 else 0

            # Update phi coefficients for next iteration
            phi_new = [0] * k
            for j in range(k-1):
                phi_new[j] = phi[j] - phi_kk * phi[k-2-j]
            phi_new[k-1] = phi_kk
            phi = phi_new

        pacf.append(phi[k-1])

    return np.array(pacf)

# Compute ACF and PACF
max_lag = 25
acf = calculate_acf(returns, max_lag)
pacf = calculate_pacf(acf, max_lag)

# 95% confidence interval
ci = 1.96 / np.sqrt(n)

# Step 5: Plot ACF and PACF
plt.figure(figsize=(12, 6))

# ACF Plot
plt.subplot(1, 2, 1)
plt.stem(range(max_lag + 1), acf, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=ci, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=-ci, color='gray', linestyle='--', alpha=0.5)
plt.title("ACF of NIFTY 50 Daily Returns")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.grid(True)

# PACF Plot
plt.subplot(1, 2, 2)
plt.stem(range(max_lag + 1), pacf, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=ci, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=-ci, color='gray', linestyle='--', alpha=0.5)
plt.title("PACF of NIFTY 50 Daily Returns")
plt.xlabel("Lag")
plt.ylabel("Partial Autocorrelation")
plt.grid(True)

plt.tight_layout()
plt.show()

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fetch training data: Jan 2015 to Dec 2023
ticker = "RELIANCE.NS"
train_data = yf.download(ticker, start="2015-01-01", end="2025-01-01", interval='1mo', progress=False)

# Fetch test data: Jan 2024 to Sep 2024 (adjusted due to current date)
test_data = yf.download(ticker, start="2025-01-01", end="2025-09-01", interval='1mo', progress=False)

# Calculate percentage returns for training data
train_data['Return'] = ((train_data['Close'] - train_data['Open']) / train_data['Open']) * 100
train_data['Return'] = train_data['Return'].fillna(method='ffill')

# Calculate percentage returns for test data
test_data['Return'] = ((test_data['Close'] - test_data['Open']) / test_data['Open']) * 100
test_data['Return'] = test_data['Return'].fillna(method='ffill')

# Prepare x and y for training
x_train = np.arange(1, len(train_data) + 1)
y_train = train_data['Return'].values
dates_train = train_data.index.strftime('%Y-%m')

# Prepare x and y for testing
x_test = np.arange(len(train_data) + 1, len(train_data) + len(test_data) + 1)
y_test = test_data['Return'].values
dates_test = test_data.index.strftime('%Y-%m')

# Compute coded x values (mean = 0)
x_all = np.concatenate([x_train, x_test])
coded_x_all = x_all - np.mean(x_train)
if len(coded_x_all)%2==0:
  coded_x_all = coded_x_all * 2
coded_x_train = coded_x_all[:len(x_train)]
coded_x_test = coded_x_all[len(x_train):]

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Linear regression using formula
def fit_linear(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    a = (sum_y - b * sum_x) / n
    return a, b

# Quadratic regression using normal equations
def fit_quadratic(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x**2)
    sum_x3 = np.sum(x**3)
    sum_x4 = np.sum(x**4)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2y = np.sum(x**2 * y)
    A = np.array([
        [n, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4]
    ])
    b_vec = np.array([sum_y, sum_xy, sum_x2y])
    coeffs = np.linalg.solve(A, b_vec)  # coeffs = [a, b, c]
    return coeffs

# Cubic regression using normal equations
def fit_cubic(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_x2 = np.sum(x**2)
    sum_x3 = np.sum(x**3)
    sum_x4 = np.sum(x**4)
    sum_x5 = np.sum(x**5)
    sum_x6 = np.sum(x**6)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2y = np.sum(x**2 * y)
    sum_x3y = np.sum(x**3 * y)
    A = np.array([
        [n, sum_x, sum_x2, sum_x3],
        [sum_x, sum_x2, sum_x3, sum_x4],
        [sum_x2, sum_x3, sum_x4, sum_x5],
        [sum_x3, sum_x4, sum_x5, sum_x6]
    ])
    b_vec = np.array([sum_y, sum_xy, sum_x2y, sum_x3y])
    coeffs = np.linalg.solve(A, b_vec)  # coeffs = [a, b, c, d]
    return coeffs


# Fit models using coded x
a_lin, b_lin = fit_linear(coded_x_train, y_train)
coeffs_quad = fit_quadratic(coded_x_train, y_train)
coeffs_cubic = fit_cubic(coded_x_train, y_train)

# Predictions on train and test sets
y_pred_lin_train = a_lin + b_lin * coded_x_train
y_pred_lin_test = a_lin + b_lin * coded_x_test
y_pred_quad_train = coeffs_quad[0] + coeffs_quad[1] * coded_x_train + coeffs_quad[2] * coded_x_train**2
y_pred_quad_test = coeffs_quad[0] + coeffs_quad[1] * coded_x_test + coeffs_quad[2] * coded_x_test**2
y_pred_cubic_train = coeffs_cubic[0] + coeffs_cubic[1] * coded_x_train + coeffs_cubic[2] * coded_x_train**2 + coeffs_cubic[3] * coded_x_train**3
y_pred_cubic_test = coeffs_cubic[0] + coeffs_cubic[1] * coded_x_test + coeffs_cubic[2] * coded_x_test**2 + coeffs_cubic[3] * coded_x_test**3

# Combine predictions
y_pred_lin = np.concatenate([y_pred_lin_train, y_pred_lin_test])
y_pred_quad = np.concatenate([y_pred_quad_train, y_pred_quad_test])
y_pred_cubic = np.concatenate([y_pred_cubic_train, y_pred_cubic_test])
y_all = np.concatenate([y_train, y_test])
dates_all = np.concatenate([dates_train, dates_test])

# Calculate RMSE on test set
rmse_lin = rmse(y_test, y_pred_lin_test)
rmse_quad = rmse(y_test, y_pred_quad_test)
rmse_cubic = rmse(y_test, y_pred_cubic_test)

print(f"Linear RMSE on test: {rmse_lin:.2f}")
print(f"Quadratic RMSE on test: {rmse_quad:.2f}")
print(f"Cubic RMSE on test: {rmse_cubic:.2f}")

# Tabulate results
def create_table(model_name, x, coded_x, y, y_pred, dates):
    df = pd.DataFrame({
        'Date': dates,
        'x (Month Index)': x,
        'Coded x': coded_x,
        'y (Returns)': y,
        f'Predicted y ({model_name})': y_pred
    })
    df['x (Month Index)'] = df['x (Month Index)'].astype(int)
    df['Coded x'] = df['Coded x'].round(2)
    df['y (Returns)'] = df['y (Returns)'].round(2)
    df[f'Predicted y ({model_name})'] = df[f'Predicted y ({model_name})'].round(2)
    print(f"\n{model_name} Regression Table:")
    print(df.to_string(index=False))

create_table("Linear", x_all, coded_x_all, y_all, y_pred_lin, dates_all)
create_table("Quadratic", x_all, coded_x_all, y_all, y_pred_quad, dates_all)
create_table("Cubic", x_all, coded_x_all, y_all, y_pred_cubic, dates_all)

# Plotting separate graphs
# Linear Regression Plot
plt.figure(figsize=(10, 5))
plt.plot(coded_x_train, y_train, 'b-', label='Training Returns')
plt.plot(coded_x_test, y_test, 'g-', label='Test Actual Returns')
plt.plot(coded_x_train, y_pred_lin_train, 'r--', label='Linear Fit')
plt.plot(coded_x_test, y_pred_lin_test, 'r-', label='Linear Prediction')
plt.axvline(x=coded_x_train[-1], color='black', linestyle='--', label='Train/Test Split')
plt.title(f"Linear Regression for AAPL Monthly Returns (RMSE: {rmse_lin:.2f})")
plt.xlabel("Coded Month Index (Centered)")
plt.ylabel("Percentage Return")
plt.legend()
plt.grid(True)
plt.show()

# Quadratic Regression Plot
plt.figure(figsize=(10, 5))
plt.plot(coded_x_train, y_train, 'b-', label='Training Returns')
plt.plot(coded_x_test, y_test, 'g-', label='Test Actual Returns')
plt.plot(coded_x_train, y_pred_quad_train, 'm--', label='Quadratic Fit')
plt.plot(coded_x_test, y_pred_quad_test, 'm-', label='Quadratic Prediction')
plt.axvline(x=coded_x_train[-1], color='black', linestyle='--', label='Train/Test Split')
plt.title(f"Quadratic Regression for AAPL Monthly Returns (RMSE: {rmse_quad:.2f})")
plt.xlabel("Coded Month Index (Centered)")
plt.ylabel("Percentage Return")
plt.legend()
plt.grid(True)
plt.show()

# Cubic Regression Plot
plt.figure(figsize=(10, 5))
plt.plot(coded_x_train, y_train, 'b-', label='Training Returns')
plt.plot(coded_x_test, y_test, 'g-', label='Test Actual Returns')
plt.plot(coded_x_train, y_pred_cubic_train, 'c--', label='Cubic Fit')
plt.plot(coded_x_test, y_pred_cubic_test, 'c-', label='Cubic Prediction')
plt.axvline(x=coded_x_train[-1], color='black', linestyle='--', label='Train/Test Split')
plt.title(f"Cubic Regression for AAPL Monthly Returns (RMSE: {rmse_cubic:.2f})")
plt.xlabel("Coded Month Index (Centered)")
plt.ylabel("Percentage Return")
plt.legend()
plt.grid(True)
plt.show()

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fetch data: Jan 2015 to Sep 2024
ticker = "RELIANCE.NS"
data = yf.download(ticker, start="2015-01-01", end="2024-10-01", interval='1mo', progress=False)

# Calculate percentage returns
data['Return'] = ((data['Close'] - data['Open']) / data['Open']) * 100
data['Return'] = data['Return'].fillna(method='ffill')

# Prepare x and y
x = np.arange(1, len(data) + 1)
y = data['Return'].values
dates = data.index.strftime('%Y-%m')

# Compute coded x values (mean = 0)
coded_x = x - np.mean(x)
if len(coded_x) % 2 == 0:
    coded_x = coded_x * 2

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

# Simple Exponential Smoothing
def simple_exponential_smoothing(y, alpha):
    smoothed = np.zeros(len(y))
    smoothed[0] = y[0]  # Initialize with first observation
    for t in range(1, len(y)):
        smoothed[t] = alpha * y[t] + (1 - alpha) * smoothed[t-1]
    return smoothed

# Generate 10 random alpha values from uniform distribution (0,1)
np.random.seed(42)  # For reproducibility
alphas = np.random.uniform(0, 1, 10)

# Tabulate and plot for each alpha
n_test = 9  # Last 9 months (Jan 2024 to Sep 2024) for RMSE evaluation
def create_table(model_name, x, coded_x, y, y_pred, dates):
    df = pd.DataFrame({
        'Date': dates,
        'x (Month Index)': x,
        'Coded x': coded_x,
        'y (Returns)': y,
        f'Smoothed y ({model_name})': y_pred
    })
    df['x (Month Index)'] = df['x (Month Index)'].astype(int)
    df['Coded x'] = df['Coded x'].round(2)
    df['y (Returns)'] = df['y (Returns)'].round(2)
    df[f'Smoothed y ({model_name})'] = df[f'Smoothed y ({model_name})'].round(2)
    print(f"\n{model_name} Table:")
    print(df.to_string(index=False))

for alpha in alphas:
    # Apply SES
    y_ses = simple_exponential_smoothing(y, alpha)

    # Calculate RMSE for last 9 months
    rmse_value = rmse(y[-n_test:], y_ses[-n_test:])

    # Print alpha and RMSE
    print(f"\nAlpha: {alpha:.2f}")
    print(f"Smoothed y values for last 9 months (Jan 2024 to Sep 2024):")
    print(y_ses[-n_test:].round(2))
    print(f"SES RMSE on last 9 months (alpha={alpha:.2f}): {rmse_value:.2f}")

    # Tabulate results
    create_table(f"SES (alpha={alpha:.2f})", x, coded_x, y, y_ses, dates)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(coded_x, y, 'b-', label='Actual Returns')
    plt.plot(coded_x, y_ses, 'r--', label='SES Smoothed')
    plt.axvline(x=coded_x[-n_test], color='black', linestyle='--', label='Last 9 Months Start')
    plt.title(f"Simple Exponential Smoothing for RELIANCE.NS Monthly Returns (RMSE: {rmse_value:.2f}, alpha={alpha:.2f})")
    plt.xlabel("Coded Month Index (Centered)")
    plt.ylabel("Percentage Return")
    plt.legend()
    plt.grid(True)
    plt.show()


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Fetch data (same as your code)
ticker = "RELIANCE.NS"
data = yf.download(ticker, start="2015-01-01", end="2024-10-01", interval='1mo', progress=False)
data['Return'] = ((data['Close'] - data['Open']) / data['Open']) * 100
data['Return'] = data['Return'].fillna(method='ffill')
y = data['Return'].values
dates = data.index.strftime('%Y-%m')

# Step 1: Check stationarity with ADF test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("Series is stationary (reject null hypothesis)")
    else:
        print("Series is non-stationary (fail to reject null hypothesis)")

print("\nADF Test for Returns:")
adf_test(y)

# Step 2: Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(y, lags=20, ax=plt.gca(), title="ACF of Returns")
plt.subplot(122)
plot_pacf(y, lags=20, ax=plt.gca(), title="PACF of Returns")
plt.tight_layout()
plt.show()

# Step 3: Define candidate ARIMA models based on ACF/PACF
# Example orders based on typical patterns (adjust after inspecting plots)
candidate_orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2)]  # ARMA models (d=0)
if adfuller(y)[1] > 0.05:
    candidate_orders = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (0, 1, 2)]  # ARIMA with d=1

# Step 4: Fit ARIMA models and compare
n_test = 9  # Last 9 months for testing
train, test = y[:-n_test], y[-n_test:]

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

best_aic = float("inf")
best_model = None
best_order = None

for order in candidate_orders:
    try:
        model = ARIMA(train, order=order).fit()
        y_pred = model.forecast(steps=n_test)
        rmse_value = rmse(test, y_pred)
        print(f"\nARIMA{order} RMSE: {rmse_value:.2f}, AIC: {model.aic:.2f}")
        if model.aic < best_aic:
            best_aic = model.aic
            best_model = model
            best_order = order
    except:
        print(f"ARIMA{order} failed to converge")

print(f"\nBest Model: ARIMA{best_order}, AIC: {best_aic:.2f}")

# Step 5: Fit best model on full data and plot
model = ARIMA(y, order=best_order).fit()
y_fitted = model.fittedvalues
y_forecast = model.forecast(steps=n_test)

# Plot actual vs fitted
x = np.arange(1, len(y) + 1)
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'b-', label='Actual Returns')
plt.plot(x, y_fitted, 'r--', label=f'ARIMA{best_order} Fitted')
plt.axvline(x=len(y) - n_test, color='black', linestyle='--', label='Last 9 Months Start')
plt.title(f"ARIMA{best_order} Fit for RELIANCE.NS Monthly Returns")
plt.xlabel("Month Index")
plt.ylabel("Percentage Return")
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Check residuals
residuals = model.resid
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(residuals)
plt.title("Residuals")
plt.subplot(122)
plot_acf(residuals, lags=20, ax=plt.gca(), title="ACF of Residuals")
plt.tight_layout()
plt.show()

# Print RMSE for last 9 months
rmse_value = rmse(y[-n_test:], model.forecast(steps=n_test))
print(f"Best ARIMA{best_order} RMSE on last 9 months: {rmse_value:.2f}")
