import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFECV
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('practice_data_germany.csv')

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

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Fit LassoCV to automatically tune the regularization
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

# Identify weak features based on low coefficients
lasso_coef = pd.Series(lasso.coef_, index=poly.get_feature_names_out(X.columns))
weak_features = lasso_coef[lasso_coef.abs() < 0.01].index  # Adjust threshold as needed

print("Weak Features Identified by Lasso:")
print(weak_features)

# Analyze feature importance with RFECV for cross-validated ranking
rfecv = RFECV(estimator=LassoCV(cv=5, random_state=42), step=1, scoring='r2')
rfecv.fit(X_scaled, y)

# Updated way to plot feature selection scores in RFECV
plt.figure()
plt.title("Feature Importance Ranking")
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (R²)")
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.show()

# Variance of original features
feature_variances = X.var().sort_values(ascending=False)
print("Feature Variances:\n", feature_variances)
