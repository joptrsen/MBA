import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load the data
df = pd.read_csv('../simulator_data.csv')

X = df.drop(columns=['Lap Time'])
y = df['Lap Time']

# Split data into train, validation, and test sets (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_val_scaler = scaler.transform(X_val)
X_test_scaler = scaler.transform(X_test)

# Hyperparameter tuning - try different polynomial degrees
best_degree = 1
best_mae = float('inf')

for degree in range(1, 5):
    # Polynomial transformation
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaler)
    X_val_poly = poly.transform(X_val_scaler)

    # Train model
    lin = LinearRegression()
    lin.fit(X_train_poly, y_train)

    # Validation predictions and MAE
    y_val_pred = lin.predict(X_val_poly)
    val_mae = mean_absolute_error(y_val, y_val_pred)

    # Check if this degree is better
    if val_mae < best_mae:
        best_mae = val_mae
        best_degree = degree

# Final model with best degree
poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train_scaler)
X_test_poly = poly.transform(X_test_scaler)

# Fit the model on the full training set
lin = LinearRegression()
lin.fit(X_train_poly, y_train)

# Predict on the training and test sets
y_pred_train = lin.predict(X_train_poly)
y_pred_test = lin.predict(X_test_poly)

# Calculate mean absolute error
print("Best Polynomial Degree:", best_degree)
print("Train MAE:", mean_absolute_error(y_train, y_pred_train))
print("Test MAE:", mean_absolute_error(y_test, y_pred_test))
print("R2:", r2_score(y_test, y_pred_test))

# Plotting the results
plt.figure(figsize=(12, 6))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Lap Time')
plt.ylabel('Predicted Lap Time')
plt.title('Training Set: Actual vs Predicted')

# Test set
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Lap Time')
plt.ylabel('Predicted Lap Time')
plt.title('Test Set: Actual vs Predicted')

plt.tight_layout()
plt.show()
