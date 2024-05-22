import gymnasium as gym
import cv2
from DQNAgent import DQNAgent

NUM_EPISODES = 30
TEST_NAME = "final_train"
MODEL = "model-700.h5"


def transform_to_grey_scale(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs.astype(float)
    obs /= 255.0
    return obs

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", render_mode='human', max_episode_steps=1000, domain_randomize=False)

    agent = DQNAgent(epsilon=0) # assure that the agent will alway stick to the policy
    agent.load_model(f"./save/{TEST_NAME}/{MODEL}")
    
    for _ in range(NUM_EPISODES):
        terminated = False
        truncated = False
        observation = transform_to_grey_scale(env.reset()[0])
        while True:
            action = agent.choose_action(observation)
            next_state, reward, done, truncated, info = env.step(action)
            observation = transform_to_grey_scale(next_state)
            if done or truncated:
                env.reset()
    env.close()