import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("spam.csv")

# Convert the numerical 'spam' column to a binary label
df['spam'] = df['spam'].apply(lambda x: 'spam' if x == 1 else 'not_spam')

# Define a ColumnTransformer to apply CountVectorizer to each column separately
preprocessor = ColumnTransformer(
    [(f'vec_{col}', CountVectorizer(), col) for col in ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']],
    remainder='drop'
)

# Create a pipeline that includes the preprocessor and Naive Bayes classifier
pipeline = Pipeline([
    ('vectorizer', preprocessor),
    ('classifier', MultinomialNB())
])

# Split the data into features (X) and target (y)
X = df[['w1', 'w2', 'w3', 'w4', 'w5', 'w6']]
y = df['spam']

# Apply cross-validation with 5 folds
cross_val_scores = cross_val_score(pipeline, X, y, cv=5)
print("Cross-validation scores:", cross_val_scores)
print(f"Mean cross-validation accuracy: {cross_val_scores.mean():.2f}")

# Training and evaluating the model on a separate test set for comparison
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"\nTest set accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# Display the first 10 actual and predicted values in a DataFrame
results_df = pd.DataFrame({
    'Actual': y_test[:10].values,
    'Predicted': y_pred[:10]
})
print("\nFirst 10 rows of actual and predicted values:")
print(results_df)
