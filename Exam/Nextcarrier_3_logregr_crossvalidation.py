import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\nextcarrier.csv')

# Preprocess the data
df.drop(['churn_dum1', 'churn_dum2'], axis=1, inplace=True)
df['age_zero'] = (df['age'] == 0).astype(int)
df['churn'] = df['churn'].astype(int)
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop(['churn'], axis=1)
y = df['churn']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Initialize logistic regression model
logreg = LogisticRegression(
    solver='lbfgs',
    max_iter=100,
    tol=0.001,
    fit_intercept=True,
    random_state=42
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
y_pred = cross_val_predict(logreg, X_scaled, y, cv=cv)

conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix (Cross-Validation):")
print(conf_matrix)

accuracy = accuracy_score(y, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y, y_pred))

# Optional: Compute AUC
y_pred_proba = cross_val_predict(logreg, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
auc = roc_auc_score(y, y_pred_proba)
print(f"AUC: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
