import numpy as np

# Constants
TOTAL_LAPS = 106
MAX_PIT_STOPS = 4
MAX_FUEL_LOAD = 80
PIT_STOP_TIME = 30  # in seconds

# Fuel Information
FUEL_CONSUMPTION_PER_LAP = 1.9542174  # (from Wear_fuelconsumption.py)
LAP_TIME_PENALTY_PER_FUEL_UNIT = 0.942887  # (from Laptime_strategyimpact.py)

# Tire wear rates per lap (out of 100) (from Wear_tyreconsumption.py)
TIRE_WEAR_RATES = {
    "Extra Soft": 10.986937667198813,
    "Soft": 3.753997684110652,
    "Medium": 2.9445815541313354,
    "Hard": 2.3573406528817737
}

# Lap time penalties for tire types (from Laptime_strategyimpact.py)
TIRE_TIME_PENALTIES = {
    "Extra Soft": 0.0,
    "Soft": 0.217923,
    "Medium": 0.112155,
    "Hard": 0.363062
}


def calculate_penalty_time(pit_strategy):
    total_penalty_time = 0
    total_fuel_penalty = 0
    total_tire_penalty = 0
    total_pit_stop_time = 0
    current_lap = 0

    for i, pit in enumerate(pit_strategy):
        pit_lap, tire_choice, fuel_load = pit

        # Determine the number of laps in this stint
        if i + 1 < len(pit_strategy):  # Check if there is a next pit stop
            next_pit_lap = pit_strategy[i + 1][0]
            stint_laps = next_pit_lap - current_lap
        else:  # If it's the last pit stop, calculate remaining laps
            stint_laps = TOTAL_LAPS - current_lap

        # Calculate penalties for the given tire choice and fuel load
        stint_fuel_penalty = 0
        stint_tire_penalty = 0

        for lap in range(stint_laps):
            # Calculate lap penalties
            fuel_penalty = fuel_load * LAP_TIME_PENALTY_PER_FUEL_UNIT
            tire_penalty = TIRE_TIME_PENALTIES[tire_choice]
            stint_fuel_penalty += fuel_penalty
            stint_tire_penalty += tire_penalty

            # Decrease fuel load after each lap
            fuel_load -= FUEL_CONSUMPTION_PER_LAP

        # Add pit stop time if this isn't the first stint
        if current_lap > 0:
            total_pit_stop_time += PIT_STOP_TIME

        # Update totals
        total_fuel_penalty += stint_fuel_penalty
        total_tire_penalty += stint_tire_penalty

        current_lap += stint_laps

        # End loop if all laps are completed
        if current_lap >= TOTAL_LAPS:
            break

    total_penalty_time = total_fuel_penalty + total_tire_penalty + total_pit_stop_time

    return total_penalty_time, total_fuel_penalty, total_tire_penalty, total_pit_stop_time



# Define your strategy here (Pit Lap, Tire Type, Fuel Load)
# Example: [(0, "Soft", 38), (17, "Soft", 38), (34, "Medium", 45), (54, "Medium", 43), (73, "Soft", 38)]
pit_strategy = [
    (0, "Hard", 64),
    (33, "Hard", 68),
    (68, "Hard", 74)
]

# Calculate penalty times based on the predefined strategy
total_penalty_time, total_fuel_penalty, total_tire_penalty, total_pit_stop_time = calculate_penalty_time(pit_strategy)

print("Best Strategy (Pit Stops):")
for pit_stop in pit_strategy:
    print(f"Pit Lap: {pit_stop[0]}, Tire: {pit_stop[1]}, Fuel Load: {pit_stop[2]:.2f}")

print("\nPenalty Time Summary:")
print(f"Total Fuel Penalty: {total_fuel_penalty:.2f} seconds")
print(f"Total Tire Penalty: {total_tire_penalty:.2f} seconds")
print(f"Total Pit Stop Time: {total_pit_stop_time:.2f} seconds")
print(f"Total Penalty Time: {total_penalty_time:.2f} seconds")
