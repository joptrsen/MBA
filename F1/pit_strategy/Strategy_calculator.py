import numpy as np
from itertools import product

# Constants
TOTAL_LAPS = 99
MAX_PIT_STOPS = 4
MAX_FUEL_LOAD = 90
MIN_FUEL_LOAD = 5
PIT_STOP_TIME = 30  # in seconds
MIN_FUEL_BEFORE_PIT = 0.000005 # Minimum remaining fuel before each pit stop


# Fuel and tire information
FUEL_CONSUMPTION_PER_LAP = 1.85255126000000558406
LAP_TIME_PENALTY_PER_FUEL_UNIT = 0.04836403975254206095
TIRE_WEAR_RATES = {
    "Extra Soft": 3.86937876,
    "Soft": 2.45789428861258,
    "Medium": 2.129366282654135,
    "Hard": 1.9197754332727273
}
TIRE_TIME_PENALTIES = {
    "Extra Soft": 0.0,
    "Soft": 0.7140333333,
    "Medium": 0.5553165000,
    "Hard": 1.0707195000
}


# Calculate stint lengths with a ±10% flexibility
def calculate_stint_lengths(num_stints):
    mean_stint = TOTAL_LAPS / num_stints
    min_stint = int(mean_stint * 0.3)
    max_stint = int(mean_stint * 1.7)



    # Generate possible stint configurations
    stint_configs = []
    for config in product(range(min_stint, max_stint + 1), repeat=num_stints):
        if sum(config) == TOTAL_LAPS:
            stint_configs.append(config)
    return stint_configs


# Calculate the penalty summary for a given pit strategy configuration
def evaluate_strategy_penalties(stints, tire_choices):
    fuel_penalty_total = 0
    tire_penalty_total = 0
    fuel_loads = []  # Store fuel load for each stint
    pit_laps = []  # Store lap numbers where each pit stop happens
    remaining_fuel_before_pit = []  # Track remaining fuel before each pit
    current_lap = 0  # Track cumulative laps to determine pit stops

    for stint_index, stint in enumerate(stints):
        tire_choice = tire_choices[stint_index]
        max_stint_length = int(100 / TIRE_WEAR_RATES[tire_choice])  # Max laps for this tire type

        if stint > max_stint_length:
            return None  # Skip strategies that exceed max stint length for the tire choice

        # Determine the number of laps to complete for this stint
        laps_to_complete = stint if stint_index == len(stints) - 1 else stint - 1

        # Calculate the required fuel load for this stint, ensuring enough fuel remains before each pit stop
        initial_fuel_load = int(np.ceil((laps_to_complete * FUEL_CONSUMPTION_PER_LAP) + MIN_FUEL_BEFORE_PIT))

        # Ensure fuel load is within limits
        if initial_fuel_load < MIN_FUEL_LOAD or initial_fuel_load > MAX_FUEL_LOAD:
            return None  # Skip infeasible fuel loads

        fuel_loads.append(initial_fuel_load)  # Track integer fuel load for this stint

        # Calculate remaining fuel at the end of the last lap before the pit stop (except for the last stint)
        if stint_index < len(stints) - 1:
            fuel_left_before_pit = initial_fuel_load - (laps_to_complete * FUEL_CONSUMPTION_PER_LAP)
            remaining_fuel_before_pit.append(fuel_left_before_pit)

        # Increment cumulative lap count and store pit lap
        current_lap += stint
        pit_laps.append(current_lap)  # This lap is where the next pit stop occurs

        # Accumulate tire penalty over the stint (fixed per lap)
        tire_penalty_total += TIRE_TIME_PENALTIES[tire_choice] * laps_to_complete

        # Calculate and accumulate fuel penalty per lap, based on decreasing fuel
        for lap in range(laps_to_complete):
            fuel_load = initial_fuel_load - (lap * FUEL_CONSUMPTION_PER_LAP)
            lap_fuel_penalty = LAP_TIME_PENALTY_PER_FUEL_UNIT * fuel_load
            fuel_penalty_total += lap_fuel_penalty

    # Final stint check: Ensure that the remaining fuel at the end of the race is above 0 liters
    final_stint_fuel_left = fuel_loads[-1] - (stints[-1] * FUEL_CONSUMPTION_PER_LAP)
    if final_stint_fuel_left <= 0:
        return None  # Skip strategies where the final stint runs out of fuel, as this would trigger an extra pit stop

    pit_stop_penalty_total = (len(stints) - 1) * PIT_STOP_TIME
    total_penalty = fuel_penalty_total + tire_penalty_total + pit_stop_penalty_total

    return total_penalty, pit_laps[:-1], fuel_loads, remaining_fuel_before_pit  # Exclude the last lap from pit_laps, which isn't a pit stop


# Find the best strategy and print each tested strategy
def find_and_print_best_strategy():
    best_total_penalty = float('inf')
    best_strategy = None

    for num_pit_stops in range(1, MAX_PIT_STOPS + 1):
        stint_configs = calculate_stint_lengths(num_pit_stops + 1)
        tire_combinations = list(
            product(TIRE_WEAR_RATES.keys(), repeat=num_pit_stops + 1))  # Generate all tire combinations

        for stints in stint_configs:
            for tire_choices in tire_combinations:
                result = evaluate_strategy_penalties(stints, tire_choices)
                if result is not None:
                    total_penalty, pit_laps, fuel_loads, remaining_fuel_before_pit = result
                    # Print each strategy
                    print(
                        f"Strategy: Pit Stops: {num_pit_stops}, Tires: {tire_choices}, Pit Laps: {pit_laps}, Total Penalty: {total_penalty:.2f} seconds")
                    print(f"Fuel Loads: {fuel_loads}, Remaining Fuel Before Pit: {remaining_fuel_before_pit}")

                    # Update the best strategy if this one has a lower total penalty
                    if total_penalty < best_total_penalty:
                        best_total_penalty = total_penalty
                        best_strategy = {
                            "num_pit_stops": num_pit_stops,
                            "tire_choices": tire_choices,
                            "pit_laps": pit_laps,
                            "fuel_loads": fuel_loads,
                            "remaining_fuel_before_pit": remaining_fuel_before_pit,
                            "total_penalty": total_penalty
                        }

    # Print the best strategy at the end
    print("\nBest Strategy:")
    print(f"Number of Pit Stops: {best_strategy['num_pit_stops']}")
    print(f"Tire Choices for Each Stint: {best_strategy['tire_choices']}")
    print(f"Laps where each pit occurs: {best_strategy['pit_laps']}")
    print(f"Fuel Loads for Each Stint: {best_strategy['fuel_loads']}")
    print(f"Remaining Fuel Before Each Pit: {best_strategy['remaining_fuel_before_pit']}")
    print(f"Total Penalty: {best_strategy['total_penalty']:.2f} seconds")


# Run the strategy search and print all strategies and the best strategy at the end
find_and_print_best_strategy()
