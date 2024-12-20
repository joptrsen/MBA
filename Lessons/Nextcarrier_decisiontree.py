import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

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

# Instantiate the decision tree model
decision_tree = DecisionTreeClassifier(random_state=42)

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
