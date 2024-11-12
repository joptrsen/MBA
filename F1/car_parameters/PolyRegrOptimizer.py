import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('simulator_data.csv')
X = df.drop(columns=['Lap Time'])
y = df['Lap Time']

# Get the first track setup as an example
example_track = [
    2.9,  # Lap Distance
    57,  # Cornering
    90,  # Inclines
    71,  # Camber
    31,  # Grip
    33,  # Temperature
    90,  # Humidity
    96,  # Air Density
    55,  # Air Pressure
    96,  # Altitude
    5,  # Roughness
    45,  # Width
    37,  # Wind (Avg. Speed)
    5   # Wind (Gusts)
]

# Define correct column groupings
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create polynomial features
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Recursive Feature Elimination with Cross-Validation (RFECV) to select optimal features
lin = LinearRegression()
rfecv = RFECV(estimator=lin, step=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
X_train_poly_rfe = rfecv.fit_transform(X_train_poly, y_train)
X_test_poly_rfe = rfecv.transform(X_test_poly)

# Train the model on the selected features
model = LinearRegression()
model.fit(X_train_poly_rfe, y_train)

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

def optimize_car_parameters(track_setup):
    initial_params = X_train[car_columns].mean().values  # Initial guesses
    param_bounds = [(1, 500) for _ in car_columns]  # Parameter bounds

    # Objective function to minimize lap time
    def objective(car_params):
        return predict_lap_time(track_setup, car_params)

    # Minimize the objective function
    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=param_bounds)
    return result.x.astype(int), result.fun

# Find optimal car parameters and predicted lap time
optimal_params, predicted_lap_time = optimize_car_parameters(example_track)

# Create results dictionary
results = {
    'Track Setup': dict(zip(track_columns, example_track)),
    'Optimal Car Parameters': dict(zip(car_columns, optimal_params)),
    'Predicted Lap Time': predicted_lap_time
}

# Print results in a formatted way
print("\nTrack Setup:")
for k, v in results['Track Setup'].items():
    print(f"{k}: {v:.2f}")

print("\nOptimal Car Parameters:")
for k, v in results['Optimal Car Parameters'].items():
    print(f"{k}: {v:.2f}")

print(f"\nPredicted Lap Time: {predicted_lap_time:.3f} seconds")

# Evaluate model performance
y_pred = model.predict(X_test_poly_rfe)
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel Mean Squared Error: {mse:.3f} seconds")

# Additional Graph: Predicted vs Actual Lap Times
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Lap Time')
plt.ylabel('Predicted Lap Time')
plt.title('Predicted vs Actual Lap Times')
plt.show()
