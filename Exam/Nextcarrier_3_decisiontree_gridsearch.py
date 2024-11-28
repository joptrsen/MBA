from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
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

param_grid = {
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [1, 11, 21, 31, 41, 51, 60, 70, 80, 90, 100],
    'min_samples_split': [4],
    'min_impurity_decrease': [0.01],
    'max_depth': [10]
}

decision_tree = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=decision_tree,
    param_grid=param_grid,
    scoring='accuracy',
    cv=10,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_imputed, y)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Cross-Validation Score: {grid_search.best_score_:.2f}')

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_imputed)

print(f'Accuracy: {accuracy_score(y, y_pred)}')
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", conf_matrix)

y_probs = best_model.predict_proba(X_imputed)[:, 1]
print(f'AUC Score: {roc_auc_score(y, y_probs):.2f}')
