import pandas as pd

import matplotlib.pyplot as plt

TEST_NAME = "alex8"


# Read the CSV file
data = pd.read_csv(f'./save/{TEST_NAME}/Training_log.csv')

# Plot the data
x_axis = data["epochs"]
y_axis = data["reward"]

plt.plot(x_axis, y_axis)





# Customize the plot if needed
plt.title('CSV Data Plot')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')

# Show the plot
plt.show()