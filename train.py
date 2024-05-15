import gymnasium as gym
import numpy as np
import cv2
from tqdm import tqdm
from DQNAgent import DQNAgent
import csv

NUM_EPISODES = 1000
BATCH_SIZE = 32
RENDERING = False
# This allow the agent to have a better perception of movement
STEP_BETWEEN_STATE = 4
TRAINING_FREQUENCY = 10
TARGET_UPDATE_FREQUENCY = 15
MODEL_SAVE_FREQUENCY = 50


def save_rewards_to_csv(episode, reward):
    with open('rewards.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if episode == 0:
            writer.writerow(["episode", "total_reward"])  # Rename the first columns
        writer.writerow([episode, total_reward])


def transform_to_grey_scale(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs.astype(float)
    obs /= 255.0
    return obs

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", max_episode_steps=300, domain_randomize=False)
    agent = DQNAgent()



    for episode in tqdm(range(NUM_EPISODES)):
        init_frame = env.reset()[0]
        current_frame = transform_to_grey_scale(init_frame)
        total_reward = 0
        negative_reward_counter = 0
        frame_counter = 0
        saved_transitions = 0 # count the number of transitions saved in the memory to only train the model every TRAINING_FREQUENCY transitions
        while(True):

                
            action = agent.choose_action(current_frame) # chose an action (either based on the model or randomly)
            
            reward = 0         
            for _ in range(STEP_BETWEEN_STATE): # it will take the same action for a few steps and observe the reward
                if RENDERING:
                    env.render()
                next_frame, r, done, truncated, info = env.step(action)
                lala = env.step(action)
                reward += r
                if done or info:
                    break
            
            if reward < 0 and frame_counter > 100:
                negative_reward_counter += 1
            else:
                negative_reward_counter = 0                
                
            # TODO: add a penalty for not moving and make it stop if the reward is too low
            # handle the reward part 
            total_reward += reward
            print(total_reward)
            if(total_reward < 0 or negative_reward_counter > 20):
                break
            
            next_frame = transform_to_grey_scale(next_frame)
            agent.save_transition(current_frame, action, reward, next_frame, done) # save the transition in the memory
            saved_transitions += 1

            
            current_frame = next_frame # update the current frame
            frame_counter += STEP_BETWEEN_STATE
            
            if len(agent.memory) > BATCH_SIZE and saved_transitions % TRAINING_FREQUENCY == 0:
                agent.replay(BATCH_SIZE) # train the model with the saved transitions
                saved_transitions = 0
            
        
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_model()
        
        if episode % MODEL_SAVE_FREQUENCY == 0:
            agent.save_model(f"./save/model-{episode}.h5")
        
        save_rewards_to_csv(episode, total_reward)
                            
           
    env.close()
