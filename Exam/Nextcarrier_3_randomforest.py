from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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

# Initialize Random Forest Classifier
random_forest = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

# Train the model
random_forest.fit(X_imputed, y)

# Predict using the model
y_pred = random_forest.predict(X_imputed)

# Evaluate model performance
print(f'Accuracy: {accuracy_score(y, y_pred)}')
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Compute predicted probabilities for AUC score
y_probs = random_forest.predict_proba(X_imputed)[:, 1]
print(f'AUC Score: {roc_auc_score(y, y_probs):.2f}')
