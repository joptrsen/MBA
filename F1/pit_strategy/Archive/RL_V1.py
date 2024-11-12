import numpy as np
import random

# Constants
TOTAL_LAPS = 90
MAX_FUEL_LOAD = 60
PIT_STOP_TIME = 30  # in seconds
FUEL_CONSUMPTION_PER_LAP = 2.28
LAP_TIME_PENALTY_PER_FUEL_UNIT = 1.200885
TIRE_WEAR_RATES = {"Extra Soft": 29.45, "Soft": 5.37, "Medium": 4.01, "Hard": 3.06}
TIRE_TIME_PENALTIES = {"Extra Soft": 0.0, "Soft": 0.398938, "Medium": 0.508485, "Hard": 0.688535}

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995
min_epsilon = 0.1
num_episodes = 1000

# Initialize Q-table (states * actions)
# For simplicity, we're using a dictionary to store Q-values for each state-action pair
Q_table = {}

def get_race_time(strategy):
    total_time = 0
    current_lap = 0

    for i, pit in enumerate(strategy):
        pit_lap, tire_choice, fuel_load = pit

        # Determine the number of laps in this stint
        if i + 1 < len(strategy):  # Check if there is a next pit stop
            next_pit_lap = strategy[i + 1][0]
            stint_laps = next_pit_lap - current_lap
        else:  # If it's the last pit stop, calculate remaining laps
            stint_laps = TOTAL_LAPS - current_lap

        # Calculate stint time for the given tire choice and fuel load
        stint_time = 0

        for lap in range(stint_laps):
            # Calculate lap time with current fuel and tire penalty
            lap_time = (fuel_load * LAP_TIME_PENALTY_PER_FUEL_UNIT) + TIRE_TIME_PENALTIES[tire_choice]
            stint_time += lap_time

            # Decrease fuel load after each lap
            fuel_load -= FUEL_CONSUMPTION_PER_LAP

        # Add pit stop time if this isn't the first stint
        if current_lap > 0:
            total_time += PIT_STOP_TIME

        # Add stint time to the total race time
        total_time += stint_time
        current_lap += stint_laps

        # End loop if all laps are completed
        if current_lap >= TOTAL_LAPS:
            break

    return total_time

def choose_action(state):
    """Choose an action based on epsilon-greedy policy."""
    if random.uniform(0, 1) < epsilon:
        # Explore: random action
        tire_choice = random.choice(list(TIRE_WEAR_RATES.keys()))
        fuel_load = random.randint(2, MAX_FUEL_LOAD)
    else:
        # Exploit: choose action with highest Q-value for the current state
        tire_choice, fuel_load = max(Q_table.get(state, {}), key=Q_table.get(state, {}).get, default=(None, None))
        if tire_choice is None:  # If no action exists, default to random
            tire_choice = random.choice(list(TIRE_WEAR_RATES.keys()))
            fuel_load = random.randint(2, MAX_FUEL_LOAD)
    return (tire_choice, fuel_load)

def update_q_table(state, action, reward, next_state):
    """Update the Q-table based on the observed reward and next state."""
    current_q = Q_table.get(state, {}).get(action, 0)
    max_future_q = max(Q_table.get(next_state, {}).values(), default=0)
    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
    if state not in Q_table:
        Q_table[state] = {}
    Q_table[state][action] = new_q

# Training
for episode in range(num_episodes):
    strategy = []
    laps_remaining = TOTAL_LAPS
    current_state = laps_remaining  # State represented by laps remaining

    while laps_remaining > 0:
        action = choose_action(current_state)  # Choose tire and fuel load
        tire_choice, fuel_load = action

        # Append action to strategy and simulate race to get reward
        strategy.append(action)
        next_state = laps_remaining - int(100 / TIRE_WEAR_RATES[tire_choice])
        next_state = max(0, next_state)  # Ensure laps remaining is not negative
        race_time = get_race_time(strategy)  # Get the race time penalty for this strategy

        reward = -race_time  # Invert race time for maximization (lower time = higher reward)
        update_q_table(current_state, action, reward, next_state)

        current_state = next_state
        laps_remaining -= int(100 / TIRE_WEAR_RATES[tire_choice])

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Output best strategy based on learned Q-table
best_strategy = max(Q_table, key=lambda state: max(Q_table[state].values()))
print("Best Strategy:", best_strategy)
