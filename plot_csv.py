import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('rewards.csv')

# Plot the data
x_axis = data["episode"]
y_axis = data["total_reward"]

plt.plot(x_axis, y_axis)





# Customize the plot if needed
plt.title('CSV Data Plot')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')

# Show the plot
plt.show()