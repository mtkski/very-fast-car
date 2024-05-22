import pandas as pd

import matplotlib.pyplot as plt

TEST_NAME = "final_train"


# Read the CSV file
data = pd.read_csv(f'./save/{TEST_NAME}/Training_log.csv')


data['time_min'] = data['time_elapsed'] / 60
# round the time to integer
data['time_min'] = data['time_min'].round(1)

#transform the memory usage collumn to integer by removing the 'MB' and converting to int
data['memory_usage'] = data['memory_usage'].str.replace(' MB', '').astype(float)

# Create the base plot
fig, ax1 = plt.subplots()

# Plot reward on the left y-axis
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Reward', color='tab:blue')
ax1.plot(data['epochs'], data['reward'], color='tab:blue', label='Reward')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for time in minutes
ax2 = ax1.twinx()
ax2.set_ylabel('epsilon', color='tab:green')
ax2.plot(data['epochs'], data['epsilon'], color='tab:green', label='epsilon')
ax2.tick_params(axis='y', labelcolor='tab:green')

# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# Show the plot
plt.title('Deep Q-Learning Performance')
plt.show()