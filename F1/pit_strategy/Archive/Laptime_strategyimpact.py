import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the data
df = pd.read_csv('../practice_data_germany.csv')

# Split the single column into separate columns based on the semicolon delimiter
df[['lap_after_pit', 'Fuel', 'Tyre Remaining', 'Tyre Choice', 'Lap Time']] = df['lap_after_pit;Fuel;Tyre Remaining;Tyre Choice;Lap Time'].str.split(';', expand=True)
df = df.drop(columns=['lap_after_pit;Fuel;Tyre Remaining;Tyre Choice;Lap Time', 'lap_after_pit', 'Tyre Remaining'])

# Convert data types
df['Fuel'] = df['Fuel'].astype(float)
df['Lap Time'] = df['Lap Time'].astype(float)

# One-hot encode the 'Tyre Choice' column
df = pd.get_dummies(df, columns=['Tyre Choice'], drop_first=True)

# Define features and target variable
X = df.drop(columns=['Lap Time'])
y = df['Lap Time']

# Polynomial Features
poly = PolynomialFeatures(degree=1, include_bias=False)
X_poly = poly.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.20, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polynomial regression model
lin = LinearRegression()

# Fit the model
lin.fit(X_train_scaled, y_train)

# Predict
y_pred_train = lin.predict(X_train_scaled)
y_pred_test = lin.predict(X_test_scaled)

# Print the intercept
print("Model Intercept:", lin.intercept_)

# Print the coefficients along with feature names
coefficients = pd.DataFrame(lin.coef_, index=poly.get_feature_names_out(X.columns), columns=['Coefficient'])
print("Model Coefficients:\n", coefficients)

# Evaluate the model with R-squared
train_r2 = lin.score(X_train_scaled, y_train)
test_r2 = lin.score(X_test_scaled, y_test)
print("\nTraining R-squared:", train_r2)
print("Test R-squared:", test_r2)

# Calculate mean absolute error and root mean squared error
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nTraining MAE:", train_mae)
print("Test MAE:", test_mae)
print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)