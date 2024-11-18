import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import numpy as np

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

# Define the cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store ROC curve values for each fold
log_reg_tprs, decision_tree_tprs = [], []
log_reg_aucs, decision_tree_aucs = [], []
mean_fpr = np.linspace(0, 1, 100)

# Logistic Regression and Decision Tree Classifiers
log_reg = LogisticRegression(solver='liblinear', random_state=42)
decision_tree = DecisionTreeClassifier(random_state=42)

# Cross-validation loop
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Logistic Regression
    log_reg.fit(X_train, y_train)
    log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, log_reg_probs)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    log_reg_tprs.append(interp_tpr)
    log_reg_aucs.append(auc(fpr, tpr))

    # Decision Tree
    decision_tree.fit(X_train, y_train)
    decision_tree_probs = decision_tree.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, decision_tree_probs)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    decision_tree_tprs.append(interp_tpr)
    decision_tree_aucs.append(auc(fpr, tpr))

# Calculate mean and std deviation of TPRs for both models
log_reg_mean_tpr = np.mean(log_reg_tprs, axis=0)
log_reg_std_tpr = np.std(log_reg_tprs, axis=0)
log_reg_mean_auc = np.mean(log_reg_aucs)
log_reg_std_auc = np.std(log_reg_aucs)

decision_tree_mean_tpr = np.mean(decision_tree_tprs, axis=0)
decision_tree_std_tpr = np.std(decision_tree_tprs, axis=0)
decision_tree_mean_auc = np.mean(decision_tree_aucs)
decision_tree_std_auc = np.std(decision_tree_aucs)

# Plotting ROC Curves
plt.figure(figsize=(10, 7))

# Plot Logistic Regression ROC curve
plt.plot(mean_fpr, log_reg_mean_tpr,
         label=f'Logistic Regression (AUC = {log_reg_mean_auc:.2f} ± {log_reg_std_auc:.2f})')
plt.fill_between(mean_fpr, log_reg_mean_tpr - log_reg_std_tpr, log_reg_mean_tpr + log_reg_std_tpr, alpha=0.2)

# Plot Decision Tree ROC curve
plt.plot(mean_fpr, decision_tree_mean_tpr,
         label=f'Decision Tree (AUC = {decision_tree_mean_auc:.2f} ± {decision_tree_std_auc:.2f})')
plt.fill_between(mean_fpr, decision_tree_mean_tpr - decision_tree_std_tpr,
                 decision_tree_mean_tpr + decision_tree_std_tpr, alpha=0.2)

# Plot a baseline for random guessing
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5)')

# Labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Logistic Regression and Decision Tree with Cross-Validation')
plt.legend(loc='lower right')
plt.show()
