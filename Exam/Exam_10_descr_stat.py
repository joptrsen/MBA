import pandas as pd

# For each customer, compute the total dollar expenditure (column “Amount”)
file_path = r'C:\Users\JoPetersen\PycharmProjects\MBA\Exam\Files\Transactions.xlsx'
df = pd.read_excel(file_path)

new_table = df.groupby('Customer ID')['Amount'].sum().reset_index()
new_table.rename(columns={'Amount': 'TotalExpenditure'}, inplace=True)

print(new_table)
