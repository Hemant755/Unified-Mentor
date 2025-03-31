import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv(r"C:\Users\theba\OneDrive\Desktop\Coffee Sale Data\index.csv", parse_dates=['date'])
print("Column Names:", df.columns)  # Print column names
print(df.head())  # Inspect first few rows
print(df.info())  # Get DataFrame info

# Strip spaces from column names
df.columns = df.columns.str.strip()
print("Stripped Column Names:", df.columns)

# Rename column if needed
df.rename(columns={'money': 'sales'}, inplace=True)
print("Renamed Column Names:", df.columns)

# Check for missing values
print("Missing Values:\n", df.isna().sum())

# Interpolate missing values
df = df.interpolate(method='linear')

# Plotting sales data as a line chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='date', y='sales')
plt.title('Daily Coffee Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['sales'], model='additive', period=365)
fig = decomposition.plot()
fig.set_size_inches(10, 6)
sns.lineplot(data=df, x='date', y='sales', alpha=0.5, label='Sales')
plt.title('Seasonal Decomposition of Coffee Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Moving Averages
df['sales_MA_7'] = df['sales'].rolling(window=7).mean()
df['sales_MA_30'] = df['sales'].rolling(window=30).mean()

# Plotting Moving Averages
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['sales'], label='Daily Sales')
plt.plot(df['date'], df['sales_MA_7'], label='7-Day Moving Average')
plt.plot(df['date'], df['sales_MA_30'], label='30-Day Moving Average')
plt.title('Daily Sales and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Creating new features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['sales_lag_1'] = df['sales'].shift(1)
df['sales_MA_7'] = df['sales'].rolling(window=7).mean()

# Drop rows with missing values in X and y
df.dropna(subset=['sales', 'sales_lag_1', 'sales_MA_7'], inplace=True)

# Splitting the data for Machine Learning
X = df[['day_of_week', 'month', 'sales_lag_1', 'sales_MA_7']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest model
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)

# Predicting and evaluating the Random Forest model
y_pred_rf = model_rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Manually calculating Root Mean Squared Error
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5

print(f'Mean Absolute Error (Random Forest): {mae_rf}')
print(f'Mean Squared Error (Random Forest): {mse_rf}')
print(f'Root Mean Squared Error (Random Forest): {rmse_rf}')

# Training a Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predicting and evaluating the Linear Regression model
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'Mean Squared Error (Linear Regression): {mse_lr}')
print(f'R^2 Score (Linear Regression): {r2_lr}')

coefficients = pd.DataFrame(model_lr.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# SQL Integration
conn = sqlite3.connect('coffee_sales.db')
df.to_sql('coffee_sales', conn, if_exists='replace', index=False)

# Query the database
query = "SELECT coffee_name, SUM(sales) as Total_Sales FROM coffee_sales GROUP BY coffee_name ORDER BY Total_Sales DESC"
result = pd.read_sql(query, conn)
print(result)

# # Export to Excel
# df.to_excel(r"C:\Users\theba\OneDrive\Desktop\Coffee Sale Data\index.xlsx", index=False)




