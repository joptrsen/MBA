import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.feature_selection import RFECV
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Example track setup
example_track = [
    3.1,  # Lap Distance
    30,   # Cornering
    55,   # Inclines
    27,   # Camber
    12,   # Grip
    17,   # Temperature
    57,   # Humidity
    34,   # Air Density
    99,   # Air Pressure
    88,   # Altitude
    24,   # Roughness
    28,   # Width
    55,   # Wind (Avg. Speed)
    35    # Wind (Gusts)
]

# Load and prepare data
df = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\F1\car_parameters\simulator_data.csv')
X = df.drop(columns=['Lap Time'])
y = df['Lap Time']

# Define track and car parameter columns
track_columns = [
    'Lap Distance', 'Cornering', 'Inclines', 'Camber', 'Grip',
    'Temperature', 'Humidity', 'Air Density', 'Air Pressure',
    'Altitude', 'Roughness', 'Width', 'Wind (Avg. Speed)', 'Wind (Gusts)'
]
car_columns = [
    'Rear Wing', 'Engine', 'Front Wing', 'Brake Balance',
    'Differential', 'Suspension'
]
all_columns = track_columns + car_columns
X = X[all_columns]  # Ensure consistent column order

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create polynomial features
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Recursive Feature Elimination with Cross-Validation (RFECV)
lin = LinearRegression()
rfecv = RFECV(estimator=lin, step=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
X_train_poly_rfe = rfecv.fit_transform(X_train_poly, y_train)
X_test_poly_rfe = rfecv.transform(X_test_poly)

# Train the model on the selected features
model = LinearRegression()
model.fit(X_train_poly_rfe, y_train)

# Additional Cross-Validation on the final model with selected features
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
cv_scores = cross_val_score(model, X_train_poly_rfe, y_train, cv=5, scoring=mse_scorer)

# Predictions and evaluation
y_pred_train = model.predict(X_train_poly_rfe)
y_pred_test = model.predict(X_test_poly_rfe)

print("Train MSE:", mean_squared_error(y_train, y_pred_train))
print("Test MSE:", mean_squared_error(y_test, y_pred_test))
print("R2 Score:", r2_score(y_test, y_pred_test))
print("Cross-Validation MSE Scores:", -cv_scores)  # Convert to positive for readability
print("Mean Cross-Validation MSE:", -cv_scores.mean())

# Custom prediction function
def predict_lap_time(track_features, car_params):
    features_dict = dict(zip(track_columns, track_features))
    features_dict.update(dict(zip(car_columns, car_params)))
    features_df = pd.DataFrame([features_dict])[all_columns]

    # Scale and transform features
    scaled_features = scaler.transform(features_df)
    poly_features = poly.transform(scaled_features)
    poly_features_rfe = rfecv.transform(poly_features)

    # Predict lap time
    return model.predict(poly_features_rfe)[0]

# Optimization function for car parameters
def optimize_car_parameters(track_setup):
    initial_params = X_train[car_columns].mean().values
    param_bounds = [(1, 500) for _ in car_columns]

    def objective(car_params):
        return predict_lap_time(track_setup, car_params)

    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=param_bounds)
    return result.x.astype(int), result.fun

# Find optimal car parameters for the example track
optimal_params, predicted_lap_time = optimize_car_parameters(example_track)

# Display results
print("\nTrack Setup:")
for k, v in dict(zip(track_columns, example_track)).items():
    print(f"{k}: {v:.2f}")

print("\nOptimal Car Parameters:")
for k, v in dict(zip(car_columns, optimal_params)).items():
    print(f"{k}: {v:.2f}")

print(f"\nPredicted Lap Time: {predicted_lap_time:.3f} seconds")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Lap Time')
plt.ylabel('Predicted Lap Time')
plt.title('Predicted vs Actual Lap Times')
plt.show()
