import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load data
df = pd.read_csv('simulator_data.csv')

# Define features and target
X = df.drop(columns=['Lap Time'])
y = df['Lap Time']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lin_reg.predict(X_test_scaled)

# Calculate and print the mean absolute error
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color="b")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
plt.xlabel("Actual Lap Time")
plt.ylabel("Predicted Lap Time")
plt.title("Actual vs. Predicted Lap Time (Test Set)")
plt.show()
