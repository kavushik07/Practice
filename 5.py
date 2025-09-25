import numpy as np
import sympy as sp
from tabulate import tabulate
import matplotlib.pyplot as plt

#Question 1)

def shift_x_values(input_years):
    avg_year = np.mean(input_years)
    shifted_years = input_years - avg_year
    if len(input_years) % 2 == 0:
        shifted_years = shifted_years * 2
    return shifted_years

def least_squares_fit(coded_x, actual_y):
    slope = np.sum(coded_x * actual_y) / np.sum(coded_x ** 2)
    intercept = np.mean(actual_y)

    x_var = sp.Symbol('x')
    regression_line = intercept + slope * x_var
    return regression_line

def trend_percentage(actual_y, predicted_y):
    percentages = []
    for act, pred in zip(actual_y, predicted_y):
        percentages.append((act / pred) * 100)
    return percentages

def cyclical_residual(actual_y, predicted_y):
    residuals = []
    for act, pred in zip(actual_y, predicted_y):
        residuals.append(((act - pred) / pred) * 100)
    return residuals

def plot_data(original_x, coded_x, actual_y, predicted_y, regression_eq):
    plt.figure(figsize=(12, 8))
    plt.plot(coded_x, actual_y, 'bo', linestyle=':', label="Observed Data", markersize=8)
    plt.plot(coded_x, predicted_y, 'go', linestyle="-", color='darkgreen', label="Fitted Line")
    plt.xticks(coded_x)
    plt.yticks(np.linspace(min(actual_y), max(actual_y), 25))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Coded Years")
    plt.ylabel("Values")
    plt.title("Trend Analysis")
    plt.show()

# Data
years = np.array([1989, 1990, 1991, 1992, 1993, 1994, 1995])
values = np.array([21.0, 19.4, 22.6, 28.2, 30.4, 24.0, 25.0])

coded_years = shift_x_values(years)
regression_equation = least_squares_fit(coded_years, values)

print("\nTrend Equation:")
print(regression_equation)
print("\n")

x_var = sp.Symbol('x')
predicted_values = [round(regression_equation.subs(x_var, val), 2) for val in coded_years]

trend_percentages = trend_percentage(values, predicted_values)
cyclic_residuals = cyclical_residual(values, predicted_values)

data_table = []
for i in range(len(years)):
    data_table.append([
        years[i],
        coded_years[i],
        values[i],
        predicted_values[i],
        round(trend_percentages[i], 2),
        round(cyclic_residuals[i], 2)
    ])

print(tabulate(data_table,
               headers=["Year", "Coded Year", "Actual Value", "Predicted Value", "Trend %", "Cyclical Residual %"],
               tablefmt="grid"))

plot_data(years, coded_years, values, predicted_values, regression_equation)

# Question 2)
# Define quarterly accounts receivable data (in thousands of dollars) from 1991 to 1995
# Sequence: Spring, Summer, Fall, Winter for each year
receivables = [102, 120, 90, 78, 110, 126, 95, 83, 111, 128, 97, 86, 115, 135, 103, 91, 122, 144, 110, 98]
total_periods = len(receivables)

time_years = [1991, 1991, 1991, 1991, 1992, 1992, 1992, 1992, 1993, 1993, 1993, 1993, 1994, 1994, 1994, 1994, 1995, 1995, 1995, 1995]
quarter_names = ['Spring', 'Summer', 'Fall', 'Winter'] * 5

# Compute 4-quarter moving averages
moving_averages = []
for i in range(total_periods - 3):
    period_sum = sum(receivables[i:i+4])  # Sum of 4 consecutive quarters
    period_avg = period_sum / 4.0         # Average over 4 quarters
    moving_averages.append(period_avg)

# Compute centered moving averages
centered_averages = []
for i in range(len(moving_averages) - 1):
    avg_centered = (moving_averages[i] + moving_averages[i+1]) / 2.0
    centered_averages.append(avg_centered)

print("\nCentered 4-Quarter Moving Averages")
for idx, avg in enumerate(centered_averages):
    period_idx = idx + 2
    year = time_years[period_idx]
    quarter = quarter_names[period_idx]
    print(f"Year {year}, {quarter}: {avg:.2f}")

# Calculate percentage of actual values to centered moving averages
actual_to_ma_ratios = []
for idx, avg in enumerate(centered_averages):
    period_idx = idx + 2
    actual_val = receivables[period_idx]
    ratio = (actual_val / avg) * 100
    actual_to_ma_ratios.append(ratio)

print("\nActual to Moving Average Percentages")
for idx, ratio in enumerate(actual_to_ma_ratios):
    period_idx = idx + 2
    year = time_years[period_idx]
    quarter = quarter_names[period_idx]
    print(f"Year {year}, {quarter}: {ratio:.2f}%")

# Group ratios by quarter type
quarter_groups = {'Spring': [], 'Summer': [], 'Fall': [], 'Winter': []}
for idx, ratio in enumerate(actual_to_ma_ratios):
    period_idx = idx + 2
    quarter = quarter_names[period_idx]
    quarter_groups[quarter].append(ratio)

# Calculate modified seasonal indices
trimmed_indices = {}
for quarter, ratios in quarter_groups.items():
    sorted_ratios = sorted(ratios)
    trimmed_ratios = sorted_ratios[1:3]  # Exclude min and max
    avg_trimmed = sum(trimmed_ratios) / len(trimmed_ratios)
    trimmed_indices[quarter] = avg_trimmed

print("\nModified Seasonal Indices")
print("Trimmed Indices:")
for quarter, idx in trimmed_indices.items():
    print(f"{quarter}: {idx:.2f}")

# Compute adjustment factor
total_trimmed = sum(trimmed_indices.values())
adj_factor = 400.0 / total_trimmed if total_trimmed != 0 else 1.0

print(f"\nAdjustment Factor: {adj_factor:.4f}")

# Calculate final seasonal indices
final_indices = {}
for quarter, idx in trimmed_indices.items():
    final_indices[quarter] = idx * adj_factor

print("\nFinal Seasonal Indices (Adjusted):")
for quarter, idx in final_indices.items():
    print(f"{quarter}: {idx:.2f}")

# Question 3)
# Define quarterly accounts receivable data (in thousands of dollars) from 1991 to 1995
# Sequence: Spring, Summer, Fall, Winter for each year
receivables = [5.6, 6.8, 6.3, 5.2, 5.7, 6.7, 6.4, 5.4, 5.3, 6.6, 6.1, 5.1, 5.4, 6.9, 6.2, 5.3]
total_periods = len(receivables)

time_years = [1992, 1992, 1992, 1992, 1993, 1993, 1993, 1993, 1994, 1994, 1994, 1994, 1995, 1995, 1995, 1995]
quarter_names = ['Spring', 'Summer', 'Fall', 'Winter'] * 4

# Compute 4-quarter moving averages
moving_averages = []
for i in range(total_periods - 3):
    period_sum = sum(receivables[i:i+4])  # Sum of 4 consecutive quarters
    period_avg = period_sum / 4.0         # Average over 4 quarters
    moving_averages.append(period_avg)

# Compute centered moving averages
centered_averages = []
for i in range(len(moving_averages) - 1):
    avg_centered = (moving_averages[i] + moving_averages[i+1]) / 2.0
    centered_averages.append(avg_centered)

print("\nCentered 4-Quarter Moving Averages")
for idx, avg in enumerate(centered_averages):
    period_idx = idx + 2
    year = time_years[period_idx]
    quarter = quarter_names[period_idx]
    print(f"Year {year}, {quarter}: {avg:.2f}")


# Create time points for plotting (align with centered moving averages)
time_points = [f"{time_years[i+2]}-{quarter_names[i+2]}" for i in range(len(centered_averages))]

# Plot original data and centered moving averages
plt.figure(figsize=(10, 6))
plt.plot(range(len(receivables)), receivables, 'bo-', label='Original Data', markersize=8, color='lightpink')
plt.plot(range(2, len(centered_averages) + 2), centered_averages, 'r*-', label='Centered Moving Average', markersize=10, color = 'lightgreen')
plt.xticks(range(len(receivables)), [f"{y}-{q[:2]}" for y, q in zip(time_years, quarter_names)], rotation=45)
plt.xlabel('Time (Year-Quarter)')
plt.ylabel('Values (Thousands of Dollars)')
plt.title('Original Data vs. Centered 4-Quarter Moving Average')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Question 4)

def shift_time_indices(time_series):
    avg_time = np.mean(time_series)
    shifted_indices = time_series - avg_time
    if len(time_series) % 2 == 0:
        shifted_indices = shifted_indices * 2
    return shifted_indices

def compute_trend_line(coded_indices, values):
    slope = np.sum(coded_indices * values) / np.sum(coded_indices ** 2)
    intercept = np.mean(values)
    time_var = sp.Symbol('t')
    trend_equation = intercept + slope * time_var
    return trend_equation

def calculate_trend_ratio(actual_vals, predicted_vals):
    ratios = [(act / pred) * 100 for act, pred in zip(actual_vals, predicted_vals)]
    return ratios

def compute_cyclic_residual(actual_vals, predicted_vals):
    residuals = [((act - pred) / pred) * 100 for act, pred in zip(actual_vals, predicted_vals)]
    return residuals

def plot_trend(original_indices, coded_indices, actual_vals, deseasonalized_vals, predicted_vals, trend_eq):
    plt.figure(figsize=(12, 8))
    plt.plot(coded_indices, actual_vals, 'go', linestyle=':', label="Observed Values", markersize=8)
    plt.plot(coded_indices, deseasonalized_vals, 'go', linestyle=':', label="Deseasonalized Values", color='purple')
    plt.plot(coded_indices, predicted_vals, 'b*-', color='blue', label="Trend Fit")
    plt.xticks(coded_indices)
    plt.yticks(np.linspace(min(actual_vals), max(actual_vals), 25))
    plt.xlabel("Coded Time Indices")
    plt.ylabel("Values")
    plt.title("Time Series Trend Analysis")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Data setup
quarterly_data = [293, 246, 231, 282, 301, 252, 227, 291, 304, 259, 239, 296, 306, 265, 240, 300]
total_quarters = len(quarterly_data)
year_list = [1992, 1992, 1992, 1992, 1993, 1993, 1993, 1993, 1994, 1994, 1994, 1994, 1995, 1995, 1995, 1995]
quarter_labels = ['Spring', 'Summer', 'Fall', 'Winter'] * 4

# Compute 4-quarter moving averages
quarterly_ma = []
for i in range(total_quarters - 3):
    ma_total = sum(quarterly_data[i:i+4])
    ma_value = ma_total / 4.0
    quarterly_ma.append(ma_value)

# Compute centered moving averages
centered_ma_values = []
for i in range(len(quarterly_ma) - 1):
    centered_value = (quarterly_ma[i] + quarterly_ma[i+1]) / 2.0
    centered_ma_values.append(centered_value)

print("\nCentered 4-Quarter Moving Averages")
for idx, value in enumerate(centered_ma_values):
    quarter_idx = idx + 2
    year = year_list[quarter_idx]
    quarter = quarter_labels[quarter_idx]
    print(f"{year} {quarter}: {value:.2f}")

# Calculate actual to moving average percentages
ma_percentages = []
for idx, ma_val in enumerate(centered_ma_values):
    quarter_idx = idx + 2
    actual_value = quarterly_data[quarter_idx]
    percentage = (actual_value / ma_val) * 100
    ma_percentages.append(percentage)

print("\nActual to Moving Average Percentages")
for idx, perc in enumerate(ma_percentages):
    quarter_idx = idx + 2
    year = year_list[quarter_idx]
    quarter = quarter_labels[quarter_idx]
    print(f"{year} {quarter}: {perc:.2f}%")

# Group percentages by quarter
quarter_groups = {'Spring': [], 'Summer': [], 'Fall': [], 'Winter': []}
for idx, perc in enumerate(ma_percentages):
    quarter_idx = idx + 2
    quarter = quarter_labels[quarter_idx]
    quarter_groups[quarter].append(perc)

# Compute modified seasonal indices
trimmed_seasonal_indices = {}
for quarter, percentages in quarter_groups.items():
    sorted_percentages = sorted(percentages)
    trimmed_values = sorted_percentages[1:3]
    avg_trimmed = sum(trimmed_values) / len(trimmed_values)
    trimmed_seasonal_indices[quarter] = avg_trimmed

print("\nModified Seasonal Indices")
print("Trimmed Seasonal Indices:")
for quarter, index in trimmed_seasonal_indices.items():
    print(f"{quarter}: {index:.2f}")

# Compute adjustment factor
total_indices_sum = sum(trimmed_seasonal_indices.values())
scaling_factor = 400.0 / total_indices_sum if total_indices_sum != 0 else 1.0
print(f"\nScaling Factor: {scaling_factor:.4f}")

# Compute final seasonal indices
adjusted_seasonal_indices = {quarter: index * scaling_factor for quarter, index in trimmed_seasonal_indices.items()}
print("\nAdjusted Seasonal Indices:")
for quarter, index in adjusted_seasonal_indices.items():
    print(f"{quarter}: {index:.2f}")

# Compute deseasonalized data
deseasonalized_values = []
for idx, value in enumerate(quarterly_data):
    quarter = quarter_labels[idx]
    seasonal_index = adjusted_seasonal_indices[quarter]
    deseasonalized = value / (seasonal_index / 100)
    deseasonalized_values.append(deseasonalized)

print("\nDeseasonalized Values")
for idx, value in enumerate(deseasonalized_values):
    year = year_list[idx]
    quarter = quarter_labels[idx]
    print(f"{year} {quarter}: {value:.2f}")

# Perform trend analysis
time_indices = np.array([i for i in range(1, len(year_list) + 1)])
coded_time_indices = shift_time_indices(time_indices)
trend_equation = compute_trend_line(coded_time_indices, deseasonalized_values)

print("\nTrend Equation:")
print(trend_equation)

time_var = sp.Symbol('t')
predicted_trend = [round(trend_equation.subs(time_var, val), 2) for val in coded_time_indices]
trend_ratios = calculate_trend_ratio(quarterly_data, predicted_trend)
cyclic_residuals = compute_cyclic_residual(quarterly_data, predicted_trend)

# Create result table
result_table = []
for i in range(len(coded_time_indices)):
    result_table.append([
        time_indices[i],
        coded_time_indices[i],
        round(deseasonalized_values[i], 2),
        predicted_trend[i],
        round(trend_ratios[i], 2),
        round(cyclic_residuals[i], 2)
    ])

print("\n" + tabulate(result_table,
                     headers=["Time", "Coded Time", "Deseasonalized", "Predicted", "Trend %", "Cyclic Residual %"],
                     tablefmt="grid"))

# Visualize results
plot_trend(time_indices, coded_time_indices, quarterly_data, deseasonalized_values, predicted_trend, trend_equation)

# Question 5)
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

# Generate business days for 2024
non_working_days = [
    '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', '2024-05-27',
    '2024-06-19', '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25'
]
business_days = []
current_date = datetime(2024, 1, 1)
while len(business_days) < 150:
    if current_date.weekday() < 5 and current_date.strftime('%Y-%m-%d') not in non_working_days:
        business_days.append(current_date.timetuple().tm_yday)
    current_date += timedelta(days=1)
trading_days = np.array(business_days, dtype=float)

# Generate synthetic stock prices
np.random.seed(42)
open_prices = np.full(150, 100.0)
daily_changes = np.random.normal(0, 1, 150)
close_prices = open_prices * (1 + daily_changes / 100)
for i in range(1, 150):
    open_prices[i] = close_prices[i-1]
    close_prices[i] = open_prices[i] * (1 + daily_changes[i] / 100)

# Perform regression analysis
def run_regression(predictors, response):
    predictors = sm.add_constant(predictors)
    regression_model = sm.OLS(response, predictors).fit()
    return regression_model

# Linear regression
X_linear = trading_days
linear_model = run_regression(X_linear, close_prices)
predicted_linear = linear_model.predict(sm.add_constant(X_linear))
residuals_linear = close_prices - predicted_linear

# Quadratic regression
X_quadratic = np.column_stack((trading_days, trading_days**2))
quadratic_model = run_regression(X_quadratic, close_prices)
predicted_quadratic = quadratic_model.predict(sm.add_constant(X_quadratic))
residuals_quadratic = close_prices - predicted_quadratic

# Cubic regression
X_cubic = np.column_stack((trading_days, trading_days**2, trading_days**3))
cubic_model = run_regression(X_cubic, close_prices)
predicted_cubic = cubic_model.predict(sm.add_constant(X_cubic))
residuals_cubic = close_prices - predicted_cubic

# Calculate R² and MSE
def calc_r_squared(actual, predicted):
    return 1 - np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2)

def calc_mse(actual, predicted):
    return np.mean((actual - predicted)**2)

r2_linear = calc_r_squared(close_prices, predicted_linear)
r2_quadratic = calc_r_squared(close_prices, predicted_quadratic)
r2_cubic = calc_r_squared(close_prices, predicted_cubic)
mse_linear = calc_mse(close_prices, predicted_linear)
mse_quadratic = calc_mse(close_prices, predicted_quadratic)
mse_cubic = calc_mse(close_prices, predicted_cubic)

print("Regression Models:")
print(f"Linear Fit: Y = {linear_model.params[0]:.3f} + {linear_model.params[1]:.5f}*t")
print(f"Quadratic Fit: Y = {quadratic_model.params[0]:.3f} + {quadratic_model.params[1]:.5f}*t + {quadratic_model.params[2]:.8f}*t²")
print(f"Cubic Fit: Y = {cubic_model.params[0]:.3f} + {cubic_model.params[1]:.5f}*t + {cubic_model.params[2]:.8f}*t² + {cubic_model.params[3]:.10f}*t³")
print("\nR² Metrics:")
print(f"Linear: {r2_linear:.4f}")
print(f"Quadratic: {r2_quadratic:.4f}")
print(f"Cubic: {r2_cubic:.4f}")
print(f"\nObservation: The cubic model has the highest R² ({r2_cubic:.4f}), indicating the best fit, but caution for potential overfitting.")
print("\nMSE Metrics:")
print(f"Linear: {mse_linear:.4f}")
print(f"Quadratic: {mse_quadratic:.4f}")
print(f"Cubic: {mse_cubic:.4f}")

# ANOVA analysis
anova_table = anova_lm(linear_model, quadratic_model, cubic_model)
print("\nANOVA Results:")
print(anova_table)

# Residual regression analysis
def analyze_residuals(residuals, predictors):
    predictors = sm.add_constant(predictors)
    residual_model = sm.OLS(residuals, predictors).fit()
    return residual_model

residual_linear_fit = analyze_residuals(residuals_linear, trading_days)
residual_quadratic_fit = analyze_residuals(residuals_quadratic, np.column_stack((trading_days, trading_days**2)))
residual_cubic_fit = analyze_residuals(residuals_cubic, np.column_stack((trading_days, trading_days**2, trading_days**3)))

print("\nResidual Regression p-values:")
print(f"Linear residuals (linear): {residual_linear_fit.pvalues[1]:.4f}")
print(f"Quadratic residuals (quadratic): {residual_quadratic_fit.pvalues[2]:.4f}")
print(f"Cubic residuals (cubic): {residual_cubic_fit.pvalues[3]:.4f}")
print("\nObservation: p-values > 0.05 suggest no significant patterns in residuals, supporting the validity of the models.")

# Visualization
plt.figure(figsize=(14, 7))
plt.scatter(trading_days, close_prices, label="Closing Prices", color="navy", alpha=0.6)
plt.plot(trading_days, predicted_linear, label="Linear Trend", color="orange", linewidth=2)
plt.plot(trading_days, predicted_quadratic, label="Quadratic Trend", color="darkgreen", linewidth=2)
plt.plot(trading_days, predicted_cubic, label="Cubic Trend", color="purple", linewidth=2)
plt.xlabel("Trading Day of Year")
plt.ylabel("Stock Price ($)")
plt.title("Stock Price Regression Analysis")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


#Quesiton 6)

import yfinance as yf
import pandas as pd
import numpy as np
import sympy as sp
from tabulate import tabulate
import matplotlib.pyplot as plt

# Fetch actual data from yfinance
stock = yf.download("GOOG", start="2020-01-01", end="2024-01-01", interval="1mo", auto_adjust=True)

# Resample quarterly
stock_q = stock['Close'].resample('QE').last()

print(stock_q)

# Quarterly returns (% change)
returns = (stock_q.pct_change() * 100).dropna()

# print(returns)

# Ensure it's a Series (avoid DataFrame error)
returns = returns.squeeze()

# Extract quarters and data
quarters = returns.index.to_period('Q').astype(str).to_list()
data = returns.tolist()

n = len(data)

# Generate years and seasons
years = [int(q[:4]) for q in quarters]
seasons = [q[4:] for q in quarters]  # 'Q1', 'Q2', etc.

def translate_x(original_x):
    average = np.mean(original_x)
    trans_x = original_x - average
    if len(original_x) % 2 == 0:
        trans_x = trans_x * 2
    return trans_x

def method_of_least_squares(x, y):
    b = np.sum(x * y) / np.sum(x ** 2)
    a = np.mean(y)
    x_sym = sp.Symbol('x')
    linear_regression_eq = a + b * x_sym
    return linear_regression_eq

def percent_of_trend_func(y_act, y_pred):
    percent = []
    for y, yp in zip(y_act, y_pred):
        percent.append((y / yp) * 100)
    return percent

def relative_cyclical_residual_func(y_act, y_pred):
    cyclical_residual = []
    for y, yp in zip(y_act, y_pred):
        cyclical_residual.append(((y - yp) / yp) * 100)
    return cyclical_residual


# ---------------- Moving Averages ----------------
ma = []
for i in range(n - 3):
    ma_sum = sum(data[i:i+4])  # Moving total for 4 quarters
    ma_avg = ma_sum / 4.0      # Divide by 4 to get moving average
    ma.append(ma_avg)

# Centered moving averages
centered_ma = []
for i in range(len(ma) - 1):
    centered = (ma[i] + ma[i+1]) / 2.0
    centered_ma.append(centered)

print("\n\n4-quarter centered moving averages")
for idx in range(len(centered_ma)):
    quarter_idx = idx + 2
    year = years[quarter_idx]
    season = seasons[quarter_idx]
    print(f"{year} {season}: {centered_ma[idx]:.2f}")

# ---------------- Percentages ----------------
percentages = []
for idx in range(len(centered_ma)):
    quarter_idx = idx + 2
    actual = data[quarter_idx]
    percent = (actual / centered_ma[idx]) * 100
    percentages.append(percent)

print("\n\nPercentage of actual to moving average")
for idx in range(len(percentages)):
    quarter_idx = idx + 2
    year = years[quarter_idx]
    season = seasons[quarter_idx]
    print(f"{year} {season}: {percentages[idx]:.2f}")

# ---------------- Seasonal Indices ----------------
season_groups = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
for idx in range(len(percentages)):
    quarter_idx = idx + 2
    season = seasons[quarter_idx]
    season_groups[season].append(percentages[idx])

print(season_groups)

modified_indices = {}
for season, percs in season_groups.items():
    sorted_percs = sorted(percs)
    if len(sorted_percs) >= 3:
        trimmed = sorted_percs[1:3]
    else:
        trimmed = sorted_percs
    modified = sum(trimmed) / len(trimmed)
    modified_indices[season] = modified

print("\n\nModified seasonal indices and seasonal indices")
print("Modified seasonal indices:")
for season, mod_idx in modified_indices.items():
    print(f"{season}: {mod_idx:.2f}")

# Adjustment factor
sum_modified = sum(modified_indices.values())
adjustment_factor = 400.0 / sum_modified if sum_modified != 0 else 1.0
print(f"\n\nAdjustment_factor : {adjustment_factor}" )

# Final seasonal indices
seasonal_indices = {}
for season, mod_idx in modified_indices.items():
    seasonal_indices[season] = mod_idx * adjustment_factor

print("\n\nSeasonal indices (adjusted):")
for season, sea_idx in seasonal_indices.items():
    print(f"{season}: {sea_idx:.2f}")

# ---------------- Deseasonalize Data ----------------
deseasonalized_data = []
for idx in range(len(data)):
    actual = data[idx]
    season = seasons[idx]
    seasonal_index = seasonal_indices[season]
    deseasonalized = actual / (seasonal_index / 100)
    deseasonalized_data.append(deseasonalized)

print("\nDeseasonalized Data")
for idx in range(len(deseasonalized_data)):
    year = years[idx]
    season = seasons[idx]
    print(f"{year} {season}: {deseasonalized_data[idx]:.2f}")

# ---------------- Trend Equation ----------------
x = [i for i in range(1, len(years)+1)]
x_coded = translate_x(x)

linear_eq = method_of_least_squares(x_coded, deseasonalized_data)

print("\nLinear regression equation\n")
print(linear_eq)
print("\n\n")

x_sym = sp.Symbol('x')
y_pred = [round(linear_eq.subs(x_sym, x_val), 2) for x_val in x_coded]

# ---------------- Trend Analysis ----------------
percent_trend_vals = percent_of_trend_func(data, y_pred)
cyclical_residual_vals = relative_cyclical_residual_func(data, y_pred)

table = []
for i in range(len(x_coded)):
    table.append([x[i], x_coded[i], data[i], deseasonalized_data[i], y_pred[i], percent_trend_vals[i], cyclical_residual_vals[i]])

print(tabulate(table, headers=["x", "Coded x", "Actual y", "Deseasonalized y", "y_pred", "Percent of trend", "Relative cyclical residual"], tablefmt="fancy_grid"))
