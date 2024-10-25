import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('simulator_data.csv')
X = df.drop(columns=['Lap Time'])
y = df['Lap Time']

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

# Ensure consistent column order
all_columns = track_columns + car_columns
X = X[all_columns]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=all_columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=all_columns,
    index=X_test.index
)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)


def predict_lap_time(track_features, car_params):
    # Create a DataFrame with all features in the correct order
    features_dict = {}
    features_dict.update(dict(zip(track_columns, track_features)))
    features_dict.update(dict(zip(car_columns, car_params)))

    features_df = pd.DataFrame([features_dict])[all_columns]

    # Scale the features
    scaled_features = pd.DataFrame(
        scaler.transform(features_df),
        columns=all_columns
    )

    # Create polynomial features
    poly_features = poly.transform(scaled_features)

    # Predict
    return model.predict(poly_features)[0]


def optimize_car_parameters(track_setup):

    # Initial car parameter values (mean values from training data)
    initial_params = X_train[car_columns].mean().values

    # Bounds for car parameters (min and max values from training data)
    param_bounds = [(1, 500) for _ in car_columns]

    # Objective function to minimize lap time
    def objective(car_params):
        return predict_lap_time(track_setup, car_params)

    # Optimize
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=param_bounds
    )

    return result.x, result.fun


# Get the first track setup as an example
example_track = X_train.iloc[0][track_columns].values

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
y_pred = model.predict(X_test_poly)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel Mean Absolute Error: {mae:.3f} seconds")