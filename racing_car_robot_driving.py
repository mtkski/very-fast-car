import gymnasium as gym
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
    agent.load_model("model.h5")
    terminated = False
    truncated = False
    observation = process_state_image(env.reset()[0])

    while not (terminated or truncated) :
        action = agent.choose_action(observation)

        observation, reward, terminated, truncated, info = env.step(action) 
        observation = process_state_image(observation)
        
        agent.train(observation, action, reward, observation, terminated)
        

        

    env.close()