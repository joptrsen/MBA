import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df1 = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\loyalty-cards-inc.csv')
df2 = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\loyalty-cards-inc-new-users.csv')

#1.	Let’s first find whether location and gender predict rewards redeemed. (We care about this variable since we obtain a commission from the retailer on each redemption). How does each variable affect predicted rewards redeemed?
'''
X = df1[['LivesNearby', 'Gender']]
y = df1['Redeemed']
reg = LinearRegression()
reg.fit(X, y)
coefficients = pd.DataFrame({
    'Predictor': ['Intercept'] + X.columns.tolist(),
    'Coefficient': [reg.intercept_] + reg.coef_.tolist()
})
print(coefficients)


#2.	Is the fit of the last regression good? What are the R-Squared, Root Mean Squared Error and the Mean Absolute Error?
r2 = reg.score(X, y)
y_pred = reg.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
'''


#3.	Please also find whether location and gender predict rewards earned. What are the conclusions here?
X = df1[['LivesNearby', 'Gender']]
y = df1['Rewards']
reg = LinearRegression()
reg.fit(X, y)
coefficients = pd.DataFrame({
    'Predictor': ['Intercept'] + X.columns.tolist(),
    'Coefficient': [reg.intercept_] + reg.coef_.tolist()
})
print(coefficients)
r2 = reg.score(X, y)
y_pred = reg.predict(X)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")


#4.	Let’s forecast the rewards redeemed of the potential loyalty card subscribers in the new sample dataset. (Note: It may be useful to know that the LivesNearby variable in the original dataset is 1 when the customer lives within 5 Km of the supermarket.)
df2['LivesNearby'] = (df2['DistanceToStore'] < 5).astype(int)
df2 = df2.drop(columns=['DistanceToStore'])
df2['PredictedRewards'] = reg.predict(df2[['LivesNearby', 'Gender']])
print(df2)


#5.	We earn 5 cents for each euro of rewards redeemed by customers. Assuming zero marketing costs and a 10% marketing effectiveness (i.e., new program subscriptions upon receiving promotional emails), what is our willingness to pay per dataset record?
