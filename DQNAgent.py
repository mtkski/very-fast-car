import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Define the AI agent
class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = [[0,0,0] , [1,0,0], [-1,0,0], [0,1,0], [0,0,0.8]]
        self.learning_rate = 1
        self.model = self.build_model()
        self.stack_size = 3
        self.stack = deque(maxlen=self.stack_size)

    def choose_action(self, observation):
        # Perform some AI logic to choose the action
        action = self.action_space[np.random.randint(0, len(self.action_space))]
        return action
    
    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, self.stack_size)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(float(self.learning_rate), epsilon=1e-7))
        return model
    
    def train(self, state, action, reward, next_state, done):
        # Convert the state and next_state to numpy arrays
        state = np.array(state)
        next_state = np.array(next_state)
        
        # Reshape the state and next_state arrays to match the input shape of the model
        state = np.reshape(state, (1, 96, 96, 1))
        next_state = np.reshape(next_state, (1, 96, 96, 1))
        
        # Predict the Q-values for the current state and next state using the model
        q_values = self.model.predict(state)[0]
        next_q_values = self.model.predict(next_state)[0]
        
        # Update the Q-value for the chosen action based on the reward and next state
        q_values[action] = reward + 0.99 * np.max(next_q_values)
        
        # Train the model using the updated Q-values
        self.model.fit(state, q_values.reshape(-1, len(self.action_space)), verbose=0)

