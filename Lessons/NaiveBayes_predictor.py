import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

# Load the training dataset
df = pd.read_csv("spam.csv")

# Convert the numerical 'spam' column to a binary label
df['spam'] = df['spam'].apply(lambda x: 'spam' if x == 1 else 'not_spam')

# Combine the word columns into a single text feature
df['text'] = df[['w1', 'w2', 'w3', 'w4', 'w5', 'w6']].agg(' '.join, axis=1)

# Split the data into features (X) and target (y)
X = df['text']
y = df['spam']

# Create a pipeline to combine vectorization and the Naive Bayes classifier
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())

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

# Load the holdout dataset
holdout_df = pd.read_csv("spam_holdout.csv")

# Combine the word columns into a single text feature for the holdout data
holdout_df['text'] = holdout_df[['w1', 'w2', 'w3', 'w4', 'w5', 'w6']].agg(' '.join, axis=1)

# Use the pipeline to predict spam and probabilities on the holdout dataset
holdout_predictions = pipeline.predict(holdout_df['text'])
holdout_probabilities = pipeline.predict_proba(holdout_df['text'])

# Create a DataFrame to display the holdout predictions with probabilities
holdout_results_df = holdout_df.copy()
holdout_results_df['Predicted_Spam'] = holdout_predictions
holdout_results_df['Probability_not_spam'] = holdout_probabilities[:, 0]  # Probability of "not_spam"
holdout_results_df['Probability_spam'] = holdout_probabilities[:, 1]      # Probability of "spam"

# Display the predictions and probabilities for the holdout dataset
print("\nPredictions and probabilities for the holdout dataset:")
print(holdout_results_df[['text', 'Predicted_Spam', 'Probability_not_spam', 'Probability_spam']])
