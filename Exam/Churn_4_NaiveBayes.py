import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the datasets
df = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\spam.csv')
df2 = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\spam_holdout.csv')

# Separate features and target variable for df
X = df[['w1', 'w2', 'w3', 'w4', 'w5', 'w6']]
y = df['spam']

# One-hot encode the categorical columns
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Initialize the Naive Bayes model with Laplace correction
model = MultinomialNB(alpha=1.0)

# Perform 10-fold cross-validation and get probabilities
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_prob = cross_val_predict(model, X_encoded, y, cv=cv, method='predict_proba')[:, 1]  # Probabilities for class 1

# Define misclassification costs
cost_fp = 1.0  # Cost of a false positive
cost_fn = 2.0  # Cost of a false negative

# Find the optimal threshold using a loop
thresholds = np.linspace(0, 1, 100)  # Test thresholds from 0 to 1
costs = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)
    costs.append(total_cost)

optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]

# Retrain the model on the full df dataset
model.fit(X_encoded, y)

# Preprocess the holdout dataset (df2)
X_holdout = df2[['w1', 'w2', 'w3', 'w4', 'w5', 'w6']]
X_holdout_encoded = encoder.transform(X_holdout)  # Use the same encoder as for df

# Predict probabilities for df2
y_holdout_prob = model.predict_proba(X_holdout_encoded)[:, 1]  # Probabilities for class 1 (spam)

# Apply the optimal threshold to classify
y_holdout_pred = (y_holdout_prob >= optimal_threshold).astype(int)

# Add predictions to df2 for inspection
df2['spam_prediction'] = y_holdout_pred

# Print results
print(f"Optimal Threshold: {optimal_threshold:.2f}")
print("Predictions for the holdout dataset (df2):")
print(df2[['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'spam_prediction']])

