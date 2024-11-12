import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the data
data = pd.read_csv('practice_data_germany.csv', delimiter=';')

# Group by Tyre Choice to analyze wear rate per lap for each type
wear_rates = {}

for tyre_type in data['Tyre Choice'].unique():
    # Filter data for each tyre type
    tyre_data = data[data['Tyre Choice'] == tyre_type]

    # Perform linear regression on lap_after_pit and Tyre Remaining
    slope, intercept, r_value, p_value, std_err = linregress(tyre_data['lap_after_pit'], tyre_data['Tyre Remaining'])

    # Store results in a dictionary
    wear_rates[tyre_type] = {
        'wear_per_lap': slope,  # Slope as absolute wear rate (change per lap)
        'intercept': intercept,
        'r_squared': r_value ** 2
    }

    # Display results
    print(f"Tyre Type: {tyre_type} - Predicted Tyre Wear per Lap: {-slope} (absolute value)")

# Plot the results
plt.figure(figsize=(10, 6))

for tyre_type in wear_rates:
    tyre_data = data[data['Tyre Choice'] == tyre_type]
    sns.scatterplot(x='lap_after_pit', y='Tyre Remaining', data=tyre_data, label=f"{tyre_type}")

    # Plot the regression line for each tire type
    plt.plot(tyre_data['lap_after_pit'],
             wear_rates[tyre_type]['intercept'] + wear_rates[tyre_type]['wear_per_lap'] * tyre_data['lap_after_pit'],
             label=f"{tyre_type} Regression Line")

plt.xlabel("Lap After Pit")
plt.ylabel("Tyre Remaining")
plt.title("Predicted Tyre Wear per Lap for Each Tyre Type")
plt.legend()
plt.show()