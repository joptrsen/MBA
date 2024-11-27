from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load train and test datasets
df1 = pd.read_excel(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\Golf.xlsx')
df2 = pd.read_excel(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\Golf-Testset.xlsx')

df1['Wind'] = df1['Wind'].replace({'true': 1, 'false': 0}).astype(int)
df1['Play'] = df1['Play'].replace({'yes': 1, 'no': 0}).astype(int)
df1 = pd.get_dummies(df1, columns=['Outlook'])
df1 = df1.astype(int)

df2['Wind'] = df2['Wind'].replace({'true': 1, 'false': 0}).astype(int)
df2['Play'] = df2['Play'].replace({'yes': 1, 'no': 0}).astype(int)
df2 = pd.get_dummies(df2, columns=['Outlook'])
df2 = df2.astype(int)

# Separate features and target for training and testing
X_train = df1.drop(['Play'], axis=1)
y_train = df1['Play']

X_test = df2.drop(['Play'], axis=1)
y_test = df2['Play']

# Impute missing values in both train and test datasets
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize Decision Tree Classifier
decision_tree = DecisionTreeClassifier(
    max_depth=10,
    criterion='gini',
    min_samples_split=4,
    min_samples_leaf=1,
    min_impurity_decrease=0.01,
    random_state=42
)

# Train the model on the training dataset
decision_tree.fit(X_train_imputed, y_train)

# Make predictions on the test dataset
y_test_pred = decision_tree.predict(X_test_imputed)

# Evaluate model performance on the test dataset
test_accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)

print(f'Test Accuracy: {test_accuracy}')
print("Confusion Matrix:\n", conf_matrix)
