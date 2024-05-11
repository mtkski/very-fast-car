import numpy as np
import tensorflow as tf
# Define the AI agent
class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = [[0,0,0] , [1,0,0], [-1,0,0], [0,1,0], [0,0,0.8]]
        self.learning_rate = 1
        self.model = self.build_model()

    def choose_action(self, observation):
        # Choose action
        observation = np.expand_dims(observation, axis=0)
        return self.action_space[np.argmax(self.model.predict(observation)[0])]
    
    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(216, activation='relu'))
        model.add(tf.keras.layers.Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(float(self.learning_rate), epsilon=1e-7))
        return model
    
    def train(self, state, action, reward, next_state, done):
        # Perform training logic
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        target = reward
        if not done:
            target = reward + 0.99 * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][np.argmax(action)] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def save_model(self, filename):
        self.model.save(filename)
    
    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)
    
    