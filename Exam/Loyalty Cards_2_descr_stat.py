import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# 1	Let’s summarize the dataset by starting with the process “Week 2”. Import it using the “File” menu.
# a)	Do most customers earn about the same amount?
# b)	Does the median customer resemble the average customer?
# c)	What is the maximum and minimum earned so far?

df = pd.read_csv(r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\loyalty-cards-inc.csv')

column_stats = df['Spend'].describe()
print(column_stats)

# 2.	What are the total rewards earned so far?
sum_rewards = df['Rewards'].sum()
print(sum_rewards)

# 3.	It could be helpful to prepare a histogram with rewards earned by customers, to be able to visualize customer heterogeneit
plt.figure(figsize=(8, 6))
plt.hist(df['Rewards'], bins=10, edgecolor='black', alpha=0.7)  # Adjust bins as needed
plt.title('Distribution of Rewards Earned by Customers')
plt.xlabel('Rewards')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 4.	Do customers of different genders earn different reward amounts?
#Version 1
stats_by_gender = df.groupby('Gender')['Rewards'].describe()
print(stats_by_gender)

# Version 2
male_rewards = df[df['Gender'] == 1]['Rewards']
female_rewards = df[df['Gender'] == 0]['Rewards']
t_stat, p_value = ttest_ind(male_rewards, female_rewards, nan_policy='omit')
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# 5.	What about different locations? Do these matter for reward earnings?
#Version 1
stats_by_location = df.groupby('LivesNearby')['Rewards'].describe()
print(stats_by_location)

# Version 2
LivesNearby_rewards = df[df['LivesNearby'] == 1]['Rewards']
LivesNotNearby_rewards = df[df['LivesNearby'] == 0]['Rewards']
t_stat, p_value = ttest_ind(LivesNearby_rewards, LivesNotNearby_rewards, nan_policy='omit')
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

