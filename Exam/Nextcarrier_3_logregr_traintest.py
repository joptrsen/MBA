import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\nextcarrier.csv')

# Preprocess the data
df.drop(['churn_dum1', 'churn_dum2'], axis=1, inplace=True)
df['age_zero'] = (df['age'] == 0).astype(int)
df['churn'] = df['churn'].astype(int)
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop(['churn'], axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

logreg = LogisticRegression(
    solver='lbfgs',
    max_iter=100,
    tol=0.001,
    fit_intercept=True,
    random_state=42
)
logreg.fit(X_train_scaled, y_train)

y_test_pred = logreg.predict(X_test_scaled)
y_test_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

# Compute and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Compute and print the accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Compute and print the AUC
auc = roc_auc_score(y_test, y_test_pred_proba)
print(f"AUC: {auc:.4f}")

# Optional: Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()