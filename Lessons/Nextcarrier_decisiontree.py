import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the decision tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(decision_tree, X, y, cv=10, scoring='accuracy')

# Print cross-validation scores and their average
print(f'Cross-Validation Scores (Accuracy): {cv_scores}')
print(f'Average Cross-Validation Score: {cv_scores.mean():.2f}')

# Compute predicted probabilities and apply the custom threshold
y_probs = decision_tree.predict_proba(X_test)[:, 1]  # Probability for the positive class (churn)
custom_threshold = 0.5  # Set your custom threshold here
y_pred_custom = (y_probs >= custom_threshold).astype(int)

# Evaluate model performance with the custom threshold
print(f'Custom Threshold ({custom_threshold}) Accuracy: {accuracy_score(y_test, y_pred_custom)}')
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)
print("Confusion Matrix with Custom Threshold:\n", conf_matrix_custom)

# Print AUC score for model evaluation
print(f'AUC Score: {roc_auc_score(y_test, y_probs):.2f}')

# Visualize the decision tree
plt.figure(figsize=(20, 10))  # Set figure size for readability
plot_tree(decision_tree, feature_names=X.columns, class_names=['Not Churn', 'Churn'], filled=True, rounded=True)
plt.show()
