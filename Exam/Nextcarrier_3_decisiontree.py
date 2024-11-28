from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

df = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\nextcarrier.csv')

df.drop(['churn_dum1', 'churn_dum2'], axis=1, inplace=True, errors='ignore')
df['age_zero'] = (df['age'] == 0).astype(int)
df['churn'] = df['churn'].astype(int)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(['churn'], axis=1)
y = df['churn']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

decision_tree = DecisionTreeClassifier(
    criterion= 'gini',
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    min_impurity_decrease=0.01,
    random_state=42
)

decision_tree.fit(X_imputed, y)
y_pred = decision_tree.predict(X_imputed)

cv_scores = cross_val_score(decision_tree, X, y, cv=10, scoring='accuracy')

print(f'Cross-Validation Scores (Accuracy): {cv_scores}')
print(f'Average Cross-Validation Score: {cv_scores.mean():.2f}')

decision_tree.fit(X, y)

y_pred = decision_tree.predict(X)

print(f'Accuracy: {accuracy_score(y, y_pred)}')
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)

y_probs = decision_tree.predict_proba(X)[:, 1]
print(f'AUC Score: {roc_auc_score(y, y_probs):.2f}')

plt.figure(figsize=(20, 10))
plot_tree(decision_tree, feature_names=X.columns, class_names=['Not Churn', 'Churn'], filled=True, rounded=True)
plt.show()
