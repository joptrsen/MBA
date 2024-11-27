from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Load dataset
df = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\nextcarrier.csv')

# Drop unnecessary columns
df.drop(['churn_dum1', 'churn_dum2'], axis=1, inplace=True, errors='ignore')

# Create new feature and ensure target is binary
df['age_zero'] = (df['age'] == 0).astype(int)
df['churn'] = df['churn'].astype(int)
df = pd.get_dummies(df, drop_first=True)

# Define features and target
X = df.drop(['churn'], axis=1)
y = df['churn']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Parameter grid for GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],  # Options for criterion
    'min_samples_leaf': [1, 11, 21, 31, 41, 51, 60, 70, 80, 90, 100],  # Leaf sizes
    'min_samples_split': [4],  # Minimum number of samples required to split
    'min_impurity_decrease': [0.01],  # Impurity decrease threshold
    'max_depth': [10]  # Maximum depth of the tree
}

# Initialize Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# GridSearchCV for parameter tuning
grid_search = GridSearchCV(
    estimator=decision_tree,
    param_grid=param_grid,
    scoring='accuracy',
    cv=10,
    n_jobs=-1,  # Use all available processors
    verbose=1   # Display progress
)

# Perform grid search
grid_search.fit(X_imputed, y)

# Display the best parameters and best score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Score: {grid_search.best_score_:.2f}')

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_imputed)

# Evaluate model performance
print(f'Accuracy: {accuracy_score(y, y_pred)}')
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Compute predicted probabilities for AUC score
y_probs = best_model.predict_proba(X_imputed)[:, 1]
print(f'AUC Score: {roc_auc_score(y, y_probs):.2f}')

# Visualize the decision tree
plt.figure(figsize=(20, 10))  # Set figure size for readability
plot_tree(best_model, feature_names=X.columns, class_names=['Not Churn', 'Churn'], filled=True, rounded=True)
plt.show()
