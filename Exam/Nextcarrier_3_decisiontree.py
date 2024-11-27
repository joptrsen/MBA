from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
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

# Initialize Decision Tree Classifier
decision_tree = DecisionTreeClassifier(
    criterion= 'gini',
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    min_impurity_decrease=0.01,
    random_state=42
)

# Train the model
decision_tree.fit(X_imputed, y)

# Make predictions
y_pred = decision_tree.predict(X_imputed)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(decision_tree, X, y, cv=10, scoring='accuracy')

# Print cross-validation scores and their average
print(f'Cross-Validation Scores (Accuracy): {cv_scores}')
print(f'Average Cross-Validation Score: {cv_scores.mean():.2f}')

# Train the decision tree on the entire dataset
decision_tree.fit(X, y)

# Predict using the default threshold of 0.5
y_pred = decision_tree.predict(X)

# Evaluate model performance
print(f'Accuracy: {accuracy_score(y, y_pred)}')
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Compute predicted probabilities for AUC score
y_probs = decision_tree.predict_proba(X)[:, 1]
print(f'AUC Score: {roc_auc_score(y, y_probs):.2f}')

# Visualize the decision tree
plt.figure(figsize=(20, 10))  # Set figure size for readability
plot_tree(decision_tree, feature_names=X.columns, class_names=['Not Churn', 'Churn'], filled=True, rounded=True)
plt.show()
