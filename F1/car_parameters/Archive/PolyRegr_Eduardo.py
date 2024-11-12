import numpy as np  # type: ignore
import statsmodels.api as sm  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.preprocessing import PolynomialFeatures  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # type: ignore
from scipy.optimize import minimize_scalar  # type: ignore
from scipy.optimize import minimize  # type: ignore

# %%
data_simulator = pd.read_csv("../simulator_data.csv")

# %%
# Predictor variables and outcome variable
X = data_simulator.drop(columns=['Lap Time'])
y = data_simulator['Lap Time']

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

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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

# %%
# Generating polynomial features
degree = 2  # Degree of polynomial features
poly_features = PolynomialFeatures(degree=degree)
X_poly_train = poly_features.fit_transform(X_train_scaled)
X_poly_test = poly_features.transform(X_test_scaled)

# %%
# Model training
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predicting on test data
y_pred = model.predict(X_poly_test)

# %%
# Calculating MSE and R2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
msee = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')
print(f'Mean Absolute Error (MAE): {mse}')


# %%
# Define predict_lap_time to use the pre-trained poly_features
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
    scaled_poly_features = poly_features.transform(scaled_features)

    # Predict
    return model.predict(scaled_poly_features)[0]


# %%
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

    return result.x, result.fun if result.success else float('inf')


# %%
# Get the first track setup as an example
example_track = X_train.iloc[0][track_columns].values

example_track = [3.6,  # Lap Distance
                 65,  # Cornering
                 82,  # Inclines
                 88,  # Camber
                 43,  # Grip
                 12,  # Temperature
                 94,  # Humidity
                 61,  # Air Density
                 34,  # Air Pressure
                 83,  # Altitude
                 20,  # Roughness
                 63,  # Width
                 51,  # Wind
                 14]  # Wind (Gust)

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
y_pred = model.predict(X_poly_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel Mean Absolute Error: {mae:.3f} seconds")
