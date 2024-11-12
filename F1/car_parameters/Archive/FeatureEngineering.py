import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../simulator_data.csv')

# Separate features and target
X = df.drop(columns=['Lap Time'])
y = df['Lap Time']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

# Polynomial transformation (degree 2 or 3 depending on complexity required)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_scaler)
X_test_poly = poly.transform(X_test_scaler)

# Recursive Feature Elimination with Cross-Validation (RFECV) with parallel processing
lin = LinearRegression()
rfecv = RFECV(estimator=lin, step=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
X_train_poly_rfe = rfecv.fit_transform(X_train_poly, y_train)
X_test_poly_rfe = rfecv.transform(X_test_poly)

# Cross-validation on the reduced feature set
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
cv_scores = cross_val_score(lin, X_train_poly_rfe, y_train, cv=5, scoring=mse_scorer)

# Fit the model on selected features
lin.fit(X_train_poly_rfe, y_train)

# Predictions for training and test sets
y_pred_train = lin.predict(X_train_poly_rfe)
y_pred_test = lin.predict(X_test_poly_rfe)

# Calculate and print MSE and R2 Score
print("Train MSE:", mean_squared_error(y_train, y_pred_train))
print("Test MSE:", mean_squared_error(y_test, y_pred_test))
print("R2 Score:", r2_score(y_test, y_pred_test))

# Print cross-validation results
print("Cross-Validation MSE Scores:", -cv_scores)  # Convert to positive for readability
print("Mean Cross-Validation MSE:", -cv_scores.mean())

# Plotting the results
plt.figure(figsize=(12, 6))

# Training set plot
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Lap Time')
plt.ylabel('Predicted Lap Time')
plt.title('Training Set: Actual vs Predicted')

# Test set plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Lap Time')
plt.ylabel('Predicted Lap Time')
plt.title('Test Set: Actual vs Predicted')

plt.tight_layout()
plt.show()
