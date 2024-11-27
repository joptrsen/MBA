import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve, mean_squared_error, r2_score
import matplotlib.pyplot as plt

#1.	Load the dataset onto RapidMiner
df = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\nextcarrier.csv')

#2.	Browse the dataset. Does everything seem generally ok with it?
#3.	Analyze the dataset via the Statistics operator. Do you see anything interesting in terms of customers’ age attribute?
descr_stat = df.describe()
#print(descr_stat)

#4.	Let’s perform the data cleaning steps required to run the logistic regression. A) Change the type of the churn variable to binomial, then set its role to the label variable. Next, remove the attributes churn_dum1 and churn_dum2. Finally, transform the age variable to take into account the issue detected in point 3 above. Paste a screenshot of the resulting process below (pc: windows key+shift+s; mac: shift+cmd+4)
df.drop(['churn_dum1', 'churn_dum2'], axis=1, inplace=True)
df['age_zero'] = (df['age'] == 0).astype(int)
df['churn'] = df['churn'].astype(int)
df = pd.get_dummies(df, drop_first=True)

#5.	Move all of the data cleaning steps into a subprocess. You can do this by selecting all of the operators and clicking the right mouse button. Select the “Move into new subprocess” option.
#6.	Add the Logistic Regression operator. Connect the Model output to the Res input to see the estimated coefficients.
#7.	Use the left “Description” option under the results tab to access additional information. Find the MSE, RMSE, R-squared.
X = df.drop(['churn'], axis=1)
y = df['churn']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

logreg = LogisticRegression(
    solver='lbfgs',
    max_iter=100,
    tol=0.001,
    fit_intercept=True,
    random_state=42
)
logreg.fit(X_scaled, y)

y_pred = logreg.predict(X_scaled)
y_pred_proba = logreg.predict_proba(X_scaled)[:, 1]

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred_proba)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared: {r2}")

conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

accuracy = accuracy_score(y, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

auc = roc_auc_score(y, y_pred_proba)
print(f"AUC: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y, y_pred))



#8.	Go  back to the design view. Use the Apply Model operator to add a prediction of the labeled variable, “churn”. A) Connect the model output of the logistic regression to the Apply Model operator. Do the same with the example output. Then, connect the outputs of the Apply Model operator to the res inputs. Can you tell how the prediction(churn) column was calculated?
df['PredictedChurn'] = y_pred
df['Confidence TP'] = y_pred_proba
df['Confidence FP'] = 1-y_pred_proba
print("\nUpdated Dataset with Predicted Churn:")
print(df)