import gymnasium as gym
import numpy as np

# Define the AI agent
class AIPlayer:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def choose_action(self, observation):
        # Perform some AI logic to choose the action
        action = (np.zeros(self.action_space.shape))
        action[0]= np.random.uniform(low=-0.3, high=0.3) #steering
        action[1]=0.5   #acceleration
        return action

def print_env_info(env):
    """
    Je vous laisse ça, rien de fou, ya sur le site, c'est juste pour capter un peu le modèle :)
    """
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(env.spec)
    print("__________________________")
    print("env.action_space = " + str(env.action_space))
    print("env.action_space.shape() = " + str(env.action_space.shape))
    print("__________________________")
    print("env.observation_space = " + str(env.observation_space))
    print("observation_space.shape = " + str(env.observation_space.shape))

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", domain_randomize=True, render_mode='human')
    print_env_info(env)

    ai_agent = AIPlayer(env.observation_space, env.action_space)
    terminated = False
    observation = env.reset()

    for i in range(100):
        action = ai_agent.choose_action(observation)
        # print(action)

        # C'est les valeurs qui sortent de step, et dans le jeu voiture on s'en fout de "info" (mais faut le mettre pour que ça marche)
        observation, reward, terminated, truncated, info = env.step(action) 

    env.close()
