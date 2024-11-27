import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load the dataset
df = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\spam.csv')

# Separate features and target variable
X = df[['w1', 'w2', 'w3', 'w4', 'w5', 'w6']]
y = df['spam']

# One-hot encode the categorical columns
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Initialize the Naive Bayes model with Laplace correction
model = MultinomialNB(alpha=1.0)

# Perform 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X_encoded, y, cv=cv, scoring='accuracy')

# Print the cross-validation results
print(f'Cross-validation scores: {scores}')
print(f'Average accuracy: {scores.mean():.2f}')
