import numpy as np
import time

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
    "Hard": 2.36
}

# Lap time penalties for tire types (from Laptime_strategyimpact.py)
TIRE_TIME_PENALTIES = {
    "Extra Soft": 0.0,
    "Soft": 0.217923,
    "Medium": 0.112155,
    "Hard": 0.363062
}

# Simulation parameters
NUM_SIMULATIONS = 200000000

def simulate_race():
    best_time = float('inf')
    best_strategy = None
    start_time = time.time()  # Start timing

    for i in range(NUM_SIMULATIONS):
        if i % (NUM_SIMULATIONS // 100) == 0 and i > 0:
            elapsed_time = time.time() - start_time
            progress = i / NUM_SIMULATIONS
            estimated_total_time = elapsed_time / progress
            remaining_time = estimated_total_time - elapsed_time
            print(f"Progress: {progress:.0%} complete - Estimated time remaining: {remaining_time / 60:.2f} minutes")

        laps_remaining = TOTAL_LAPS
        total_time = 0
        current_lap = 0
        pit_strategy = []

        while laps_remaining > 0:
            tire_choice = np.random.choice(list(TIRE_WEAR_RATES.keys()))
            max_stint_laps = int(100 / TIRE_WEAR_RATES[tire_choice])
            stint_laps = min(np.random.randint(2, max_stint_laps + 1), laps_remaining)

            required_fuel = int(min(stint_laps * FUEL_CONSUMPTION_PER_LAP, MAX_FUEL_LOAD))
            fuel_load = required_fuel
            stint_time = 0

            for lap in range(stint_laps):
                lap_time = (fuel_load * LAP_TIME_PENALTY_PER_FUEL_UNIT) + TIRE_TIME_PENALTIES[tire_choice]
                stint_time += lap_time
                fuel_load -= FUEL_CONSUMPTION_PER_LAP

            if current_lap > 0:
                total_time += PIT_STOP_TIME

            total_time += stint_time
            pit_strategy.append((current_lap, tire_choice, required_fuel))

            laps_remaining -= stint_laps
            current_lap += stint_laps

            if laps_remaining <= 0:
                break

        if total_time < best_time:
            best_time = total_time
            best_strategy = pit_strategy

    return best_strategy, best_time


# Run simulation with continuous time updates
best_strategy, best_race_time = simulate_race()

# Display final results
print("Best Strategy (Pit Stops):")
for pit_stop in best_strategy:
    print(f"Pit Lap: {pit_stop[0]}, Tire: {pit_stop[1]}, Fuel Load: {pit_stop[2]:.2f}")
print("Total penalty Time:", best_race_time)
