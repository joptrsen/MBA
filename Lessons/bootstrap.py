import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data without a header
data = pd.read_excel('bootstrap.xlsx', header= 0)  # Use header=None to indicate no header row

# Prepare the data
X = data[['Credit score']]  # Feature as a DataFrame (2D)
y = data['Amount owed']    # Target as a Series (1D)

# Split data into training and test sets for better evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Print the coefficients and evaluation metrics
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Example prediction
example_amount = [[8000]]  # Example input for Amount Owed
predicted_score = model.predict(example_amount)
print(f"Predicted Credit Score for Amount Owed = 8000: {predicted_score[0]}")
