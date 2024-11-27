import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_excel(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\Titanic.xlsx')

df = df.drop(['Name', 'Ticket Number'], axis=1)

df['no_lifeboat'] = df['Life Boat'].isnull().astype(int)
df['no_cabin'] = df['Cabin'].isnull().astype(int)
df['no_age'] = df['Age'].isnull().astype(int)

df['Survived'] = df['Survived'].replace({'Yes': 1, 'No': 0}).astype(int)

string_columns = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=string_columns, drop_first=True)

X = df.drop(['Survived'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

max_depth_values = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

results = []

for max_depth in max_depth_values:
    decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    decision_tree.fit(X_train, y_train)

    y_test_pred = decision_tree.predict(X_test)
    y_train_pred = decision_tree.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    results.append({'max_depth': max_depth, 'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy})

accuracy_table = pd.DataFrame(results)

print(accuracy_table)
