import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc

# Load the dataset
df = pd.read_csv('nextcarrier.csv')

# Drop unnecessary columns and filter out rows where 'age' is 0
df = df.drop(columns=['churn_dum1', 'churn_dum2'])
df = df.loc[df['age'] != 0]

# One-Hot Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Prepare features and target variable
X = df.drop('churn', axis=1)
y = df['churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the logistic regression model
log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(X_train, y_train)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(log_reg, X, y, cv=10, scoring='accuracy')

# Print cross-validation scores and their average
print(f'Cross-Validation Scores (Accuracy): {cv_scores}')
print(f'Average Cross-Validation Score: {cv_scores.mean():.2f}')

'''
# View intercept and coefficients
print(f'Intercept: {log_reg.intercept_}')
coeff_df = pd.DataFrame(log_reg.coef_.flatten(), index=X.columns, columns=['Coefficient'])
print(coeff_df)
'''

# Predict churn using the default threshold (for comparison)
y_pred_default = log_reg.predict(X_test)
print(f'Default Threshold Accuracy: {accuracy_score(y_test, y_pred_default)}')
conf_matrix_default = confusion_matrix(y_test, y_pred_default)
print("Confusion Matrix with Default Threshold:\n", conf_matrix_default)
#print("Classification Report with Default Threshold:\n", classification_report(y_test, y_pred_default))
'''
# Plot confusion matrix for default threshold
sns.heatmap(conf_matrix_default, annot=True, fmt='d')
plt.title('Confusion Matrix (Default Threshold = 0.5)')
plt.show()
'''
# Compute predicted probabilities and ROC curve
y_probs = log_reg.predict_proba(X_test)[:, 1]  # Select the probability for the positive class (churn)
'''
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random performance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
'''
# Print AUC score
print(f'AUC Score: {roc_auc_score(y_test, y_probs):.2f}')

# Apply a custom threshold to the probabilities
custom_threshold = 0.5 # Set your custom threshold here
y_pred_custom = (y_probs >= custom_threshold).astype(int)

# Evaluate model performance with the custom threshold
print(f'Custom Threshold ({custom_threshold}) Accuracy: {accuracy_score(y_test, y_pred_custom)}')
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)
print("Confusion Matrix with Custom Threshold:\n", conf_matrix_custom)
#print("Classification Report with Custom Threshold:\n", classification_report(y_test, y_pred_custom))
'''
# Plot confusion matrix for the custom threshold
sns.heatmap(conf_matrix_custom, annot=True, fmt='d')
plt.title(f'Confusion Matrix (Custom Threshold = {custom_threshold})')
plt.show()

# Create a DataFrame to analyze predictions with their confidence levels
results_df = X_test.copy()
results_df['True_Label'] = y_test.values
results_df['Predicted_Label'] = y_pred_custom
results_df['Prediction_Confidence'] = y_probs
results_df['Correct_Prediction'] = results_df['True_Label'] == results_df['Predicted_Label']

# Separate true and false predictions with their confidence
true_predictions = results_df[results_df['Correct_Prediction'] == True]
false_predictions = results_df[results_df['Correct_Prediction'] == False]

# Display the confidence levels for both true and false predictions
print("True Predictions with Confidence:")
print(true_predictions[['True_Label', 'Predicted_Label', 'Prediction_Confidence']].head())

print("\nFalse Predictions with Confidence:")
print(false_predictions[['True_Label', 'Predicted_Label', 'Prediction_Confidence']].head())
'''