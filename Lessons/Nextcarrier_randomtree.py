import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc

# Load the dataset
df = pd.read_csv('nextcarrier.csv')

# Drop unnecessary columns and filter out rows where 'age' is 0
df = df.drop(columns=['churn_dum1', 'churn_dum2'])
df = df.loc[df['age'] != 0]

# One-Hot Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Prepare features and target variable
X = df.drop('churn', axis=1)
y = df['churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Instantiate the RandomForestClassifier
random_forest = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score from grid search
print("Best Parameters from Grid Search:", grid_search.best_params_)
print(f"Best AUC Score from Grid Search: {grid_search.best_score_:.2f}")

# Use the best estimator from grid search to make predictions
best_rf_model = grid_search.best_estimator_
y_probs = best_rf_model.predict_proba(X_test)[:, 1]

# Evaluate the optimized model
print(f'AUC Score (Optimized Model): {roc_auc_score(y_test, y_probs):.2f}')

# Plot ROC Curve for the optimized model
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'Optimized ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Optimized Random Forest Model')
plt.legend(loc='lower right')
plt.show()
