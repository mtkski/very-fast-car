import gymnasium as gym
import cv2
from DQNAgent import DQNAgent

NUM_EPISODES = 30

def transform_to_grey_scale(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs.astype(float)
    obs /= 255.0
    return obs

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode='human', max_episode_steps=300)

    agent = DQNAgent(epsilon=0) # assure that the agent will alway stick to the policy
    agent.load_model("./save/model-50.h5")
    
    for _ in range(NUM_EPISODES):
        terminated = False
        truncated = False
        observation = transform_to_grey_scale(env.reset()[0])
        while True:
            action = agent.choose_action(observation)
            next_state, reward, done, truncated, info = env.step(action)
            observation = transform_to_grey_scale(next_state)
            if done or truncated:
                break
        


        
        

        

    env.close()