import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('practice_data_germany.csv', delimiter=';')

# Set display options to avoid scientific notation
pd.options.display.float_format = '{:.8f}'.format

data['Fuel_abs_change'] = data['Fuel'].diff()
data['Fuel_abs_change'] = data['Fuel_abs_change'].apply(lambda x: x if x < 0 else 0)

# Filter data to exclude points where lap_after_pit is less than or equal to 2
filtered_data = data[data['lap_after_pit'] > 1]

# Calculate the average Fuel_abs_change
average_fuel_abs_change = filtered_data['Fuel_abs_change'].mean()

# Calculate variance of Fuel_abs_change
variance_fuel_abs_change = filtered_data['Fuel_abs_change'].var()

# Print variance
print(f"Variance of Fuel_abs_change: {variance_fuel_abs_change}")

# Get descriptive statistics and include variance
fuel_abs_change_stats = filtered_data['Fuel_abs_change'].describe()
fuel_abs_change_stats['variance'] = variance_fuel_abs_change
print("\nDescriptive statistics for Fuel_abs_change in filtered data:")
print(fuel_abs_change_stats)

# Plot to visually inspect the correlation with color by Tyre Choice
plt.figure(figsize=(10, 6))
sns.scatterplot(x='lap_after_pit', y='Fuel_abs_change', data=filtered_data, hue='Tyre Choice', palette="viridis", s=100)

# Add a horizontal line for the average Fuel_abs_change
plt.axhline(average_fuel_abs_change, color='blue', linestyle='--', label=f"Average Fuel Change: {average_fuel_abs_change}")

# Labels and title
plt.xlabel("Lap After Pit")
plt.ylabel("Fuel Consumption")
plt.title("Correlation between Lap After Pit and Fuel Consumption with Average Fuel Change")
plt.legend()
plt.show()
