import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('practice_data_portugal.csv')

# Split the data into separate columns based on the semicolon delimiter
df[['lap_after_pit', 'Fuel', 'Tyre Remaining', 'Tyre Choice', 'Lap Time']] = df[
    'lap_after_pit;Fuel;Tyre Remaining;Tyre Choice;Lap Time'].str.split(';', expand=True)
df = df.drop(columns=['lap_after_pit;Fuel;Tyre Remaining;Tyre Choice;Lap Time', 'Fuel', 'Tyre Remaining'])

# Convert necessary columns to numeric types
df['lap_after_pit'] = df['lap_after_pit'].astype(int)
df['Lap Time'] = df['Lap Time'].astype(float)

# Separate Extra Soft lap times for each lap after pit
extra_soft_times = df[df['Tyre Choice'] == 'Extra Soft'][['lap_after_pit', 'Lap Time']].rename(
    columns={'Lap Time': 'Extra Soft Lap Time'})

# Merge each tire's lap times with the Extra Soft lap times on lap_after_pit
df = df.merge(extra_soft_times, on='lap_after_pit', how='left')

# Calculate the lap time difference relative to Extra Soft for each lap after pit
df['Lap Time Difference to Extra Soft'] = df['Lap Time'] - df['Extra Soft Lap Time']

# Group by tire type to calculate mean and median differences
mean_diff_by_tire = df.groupby('Tyre Choice')['Lap Time Difference to Extra Soft'].mean()
median_diff_by_tire = df.groupby('Tyre Choice')['Lap Time Difference to Extra Soft'].median()

# Print the mean and median differences with 10 decimal places
print("Mean Lap Time Difference to Extra Soft by Tire Type (10 decimal places):")
print(mean_diff_by_tire.apply(lambda x: f"{x:.10f}"))

print("\nMedian Lap Time Difference to Extra Soft by Tire Type (10 decimal places):")
print(median_diff_by_tire.apply(lambda x: f"{x:.10f}"))

# Plotting the boxplot for lap time differences to Extra Soft by tire type
plt.figure(figsize=(10, 6))
plt.title("Lap Time Difference to Extra Soft by Tire Type (Lap-by-Lap)")
plt.xlabel("Tire Type")
plt.ylabel("Lap Time Difference to Extra Soft")

# Create boxplot for lap time differences by tire type
df.boxplot(column='Lap Time Difference to Extra Soft', by='Tyre Choice', grid=False)

# Annotate the mean and median differences for each tire type on the plot with 10 decimal places
for tick, tyre_type in enumerate(mean_diff_by_tire.index, 1):
    mean_diff = mean_diff_by_tire[tyre_type]
    median_diff = median_diff_by_tire[tyre_type]

    # Plot mean as a red dot and annotate mean and median with 10 decimal places
    plt.plot(tick, mean_diff, 'ro')  # 'ro' makes red dots for the mean values
    plt.text(tick, mean_diff, f'{mean_diff:.2f}', color='red', ha='center', va='bottom', fontsize=9)
    plt.text(tick, median_diff, f'{median_diff:.2f}', color='blue', ha='center', va='top', fontsize=9)

plt.suptitle("")  # Remove default title to keep only the main title
plt.show()
