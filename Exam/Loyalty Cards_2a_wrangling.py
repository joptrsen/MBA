import pandas as pd

df1 = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\loyalty-cards-inc.csv')
df2 = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\loyalty-cards-inc-visits.csv')

# 1.	Let’s introduce a new column calculating the percent rewards redeemed so far.
df1['percentage_redeemed'] = df1['Redeemed'] / df1['Rewards']

# 2.	Let’s focus only on customers who spent at least 200 euros in the store.
filtered_data1 = df1[df1['Spend'] > 200]

# 3.	GDPR is out there. Let’s get rid of customer names
filtered_data2 = filtered_data1.drop(['Name'], axis=1)

# 4.	Merge the dataset with file loyalty-cards-inc-visits.csv. This will enable us to add a column containing the number of customer visits to our store.
merged_data = filtered_data2.merge(df2, on='ID', how='inner')

# 5.	Create new unique ID variable and scrap the old one. Make sure the new variable is named “ID”, not “id”. (This is often useful when customer id’s are not consecutive, for example.)
merged_data = filtered_data1.drop(['ID'], axis=1)
merged_data['ID'] = range(1, len(merged_data) + 1)

# 6.	Finally, create a new variable that contains the average spend for the person’s gender. This means all people from each gender will have the same value in this variable.
average_spend_by_gender = merged_data.groupby('Gender')['Spend'].transform('mean')
merged_data['AvgSpendByGender'] = average_spend_by_gender
print(merged_data.head())
