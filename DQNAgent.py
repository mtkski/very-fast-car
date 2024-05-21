import numpy as np
import random
import tensorflow as tf
from collections import deque
# Define the AI agent
class DQNAgent:
    def __init__(
        self,
        action_space = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 0.8)],
        memory_size=5000,
        learning_rate=0.001,
        epsilon=1.0,
        frame_window_size=2
    ):
        self.action_space = action_space
        self.memory_size = memory_size
        # create a list of max memory size
        self.memory = deque(maxlen=memory_size)
        self.frame_window_size = frame_window_size
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        
        

    def choose_action(self, observation):
        # Choose action based on epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        else:
            prediction = self.model.predict(np.expand_dims(observation, axis=0))[0] # get the prediction of the model (list of Q values for each action)
            action_index = np.argmax(prediction) # get the index of the action with the highest Q value
            return self.action_space[action_index]
    
    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = tf.keras.models.Sequential()
        init = tf.keras.initializers.HeUniform()
        model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(self.frame_window_size, 96, 96), kernel_initializer=init))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=12, kernel_size=(4, 4), activation='relu', kernel_initializer=init))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(216, activation='relu', kernel_initializer=init))
        model.add(tf.keras.layers.Dense(len(self.action_space), activation=None, kernel_initializer=init))
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(float(self.learning_rate), epsilon=1e-7))
        return model

    def save_transition(self, current_state_window, action, reward, next_state_window, done):
        # Save transition to replay memory
        self.memory.append((current_state_window, action, reward, next_state_window, done))
    
    def replay(self, batch_size):
        # Train model
        batch = random.sample(self.memory, batch_size)
        train_states = []
        train_targets = []
        for state, action, reward, next_state, done in batch:
            action_index = self.action_space.index(action)
            
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action_index] = reward
            else:
                Q_future = max(self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
                target[action_index] = reward + Q_future * self.gamma
            train_states.append(state)
            train_targets.append(target)
        self.model.fit(np.array(train_states), np.array(train_targets), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, filename):
        self.model.save(filename)
    
    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)
        self.target_model = tf.keras.models.load_model(filename)
    
    