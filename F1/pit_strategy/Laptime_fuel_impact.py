import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('practice_data_portugal.csv')

# Split the data into separate columns based on the semicolon delimiter
df[['lap_after_pit', 'Fuel', 'Tyre Remaining', 'Tyre Choice', 'Lap Time']] = df[
    'lap_after_pit;Fuel;Tyre Remaining;Tyre Choice;Lap Time'].str.split(';', expand=True)
df = df.drop(columns=['lap_after_pit;Fuel;Tyre Remaining;Tyre Choice;Lap Time', 'Tyre Remaining'])

# Convert necessary columns to numeric types
df['lap_after_pit'] = df['lap_after_pit'].astype(int)
df['Fuel'] = df['Fuel'].astype(float)
df['Lap Time'] = df['Lap Time'].astype(float)

# Create a 'stint_id' column to identify each stint
df['stint_id'] = (df['lap_after_pit'] == 1).cumsum()

# Calculate the lap time change per unit of fuel within each stint, excluding the grouping columns
df['Lap Time Change per Fuel Unit'] = df.groupby('stint_id', group_keys=False).apply(
    lambda stint: stint['Lap Time'].diff() / stint['Fuel'].diff()
)

# Drop NaN values resulting from diff() operations within stints
df.dropna(subset=['Lap Time Change per Fuel Unit'], inplace=True)

# Save the cleaned DataFrame to a new Excel file
df.to_excel('cleaned_data.xlsx', index=False)

# Calculate the overall mean and median for lap time change per fuel unit across all tire types
overall_mean = df['Lap Time Change per Fuel Unit'].mean()
overall_median = df['Lap Time Change per Fuel Unit'].median()

# Print the overall mean and median values
print(f"Overall Mean Lap Time Change per Fuel Unit: {overall_mean:.20f}")
print(f"Overall Median Lap Time Change per Fuel Unit: {overall_median:.20f}")

# Plotting the distribution of lap time change per fuel unit for each tire type, with mean and median annotations
plt.figure(figsize=(10, 6))
plt.title("Boxplot of Lap Time Change per Fuel Unit by Tire Type with Mean and Median Values")
plt.xlabel("Tire Type")
plt.ylabel("Lap Time Change per Fuel Unit")

# Create the boxplot
boxplot = df.boxplot(column='Lap Time Change per Fuel Unit', by='Tyre Choice', grid=False)

# Calculate mean and median values by tire type for individual annotations
means = df.groupby('Tyre Choice')['Lap Time Change per Fuel Unit'].mean()
medians = df.groupby('Tyre Choice')['Lap Time Change per Fuel Unit'].median()

# Annotate individual mean and median for each tire type
for tick, tyre_type in enumerate(means.index, 1):
    mean = means[tyre_type]
    median = medians[tyre_type]

    # Plot mean as a red dot and median text annotation
    plt.plot(tick, mean, 'ro')  # 'ro' makes red dots for the mean values
    plt.text(tick, mean, f'{mean:.2f}', color='red', ha='center', va='bottom', fontsize=9)
    plt.text(tick, median, f'{median:.2f}', color='blue', ha='center', va='top', fontsize=9)

# Add overall mean and median as horizontal lines across the plot
plt.axhline(y=overall_mean, color='red', linestyle='--', label=f'Overall Mean: {overall_mean:.2f}')
plt.axhline(y=overall_median, color='blue', linestyle='--', label=f'Overall Median: {overall_median:.2f}')

# Add legend for the overall lines
plt.legend()

plt.suptitle("")  # Remove default title to keep only the main title
plt.show()
