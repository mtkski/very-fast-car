import gymnasium as gym
import numpy as np
import cv2
from DQNAgent import DQNAgent

def process_state_image(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs.astype(float)
    obs /= 255.0
    return obs

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode='human', max_episode_steps=300)

    agent = DQNAgent(env.observation_space, env.action_space)
    terminated = False
    truncated = False
    current_observation = process_state_image(env.reset()[0]) # Initial observation
    agent.stack.extend([current_observation] * agent.stack_size) 
    deque([init_state]*agent.frame_stack_num, maxlen=agent.frame_stack_num)

    while not (terminated or truncated) :
        action = agent.choose_action()

        current_observation, reward, terminated, truncated, info = env.step(action) 
        current_observation = process_state_image(current_observation)
        agent.stack.append(current_observation)

    env.close()
