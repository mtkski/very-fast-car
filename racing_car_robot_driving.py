import gymnasium as gym
import cv2
from DQNAgent import DQNAgent
from collections import deque
import numpy as np

NUM_EPISODES = 30
TEST_NAME = "alex8"
MODEL = "model-700.h5"


def transform_to_grey_scale(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs.astype(float)
    obs /= 255.0
    return obs

def adapt_queu_shape_to_model_input(queue):
    return np.array(queue).reshape(96, 96, len(queue))

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode='human', max_episode_steps=1000, domain_randomize=False)

    agent = DQNAgent(epsilon=0) # assure that the agent will alway stick to the policy
    agent.load_model(f"./save/{TEST_NAME}/{MODEL}")
    
    for _ in range(NUM_EPISODES):
        terminated = False
        truncated = False
        observation = transform_to_grey_scale(env.reset()[0])
        frame_window = deque([observation]*agent.frame_window_size, maxlen=agent.frame_window_size)
        while True:
            action = agent.choose_action(adapt_queu_shape_to_model_input(frame_window))
            next_state, reward, done, truncated, info = env.step(action)
            observation = transform_to_grey_scale(next_state)
            frame_window.append(observation)
        
            if done or truncated:
                env.reset()
    env.close()