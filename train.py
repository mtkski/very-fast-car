import gymnasium as gym
import numpy as np
import cv2
from tqdm import tqdm
from DQNAgent import DQNAgent
import csv
from collections import deque
import psutil
from memory_profiler import profile
import time



TEST_NAME = "final_train"
NUM_EPOCS = 1000
STARTING_EPOCH = 0
BATCH_SIZE = 32
RENDERING = False
# This allow the agent to have a better perception of movement
STEP_BETWEEN_STATE = 4
TRAINING_FREQUENCY = 10
TARGET_UPDATE_FREQUENCY = 15
MODEL_SAVE_FREQUENCY = 20
MIN_FRAME_FOR_NEG_REWARD = 50
MAX_CONSECUTIVE_NEG_REWARD = 5
LOADING_EXISTING_MODEL = False
LOADING_MODEL_PATH = f"./save/{TEST_NAME}/model-220.h5"


def save_log(epochs, reward, agent, memory_usage, time_elapsed):
    with open(f'./save/{TEST_NAME}/Training_log.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epochs % 10 == 0:
            if csvfile.tell() == 0:
                writer.writerow(["test_name", "epochs", "reward", "epsilon", "epsilon_min", "espilon_decay ", "gamma" , "memory_size" , "BATCH_SIZE size", "STEP_BETWEEN_STATE", "TRAINING_FREQUENCY", "TARGET_UPDATE_FREQUENCY", "MIN_FRAME_FOR_NEG_REWARD", "memory_usage", "time_elapsed"])  # Rename the first columns
                writer.writerow([TEST_NAME, epochs, np.max(reward), agent.epsilon , agent.epsilon_min, agent.epsilon_decay, agent.gamma , agent.memory_size, BATCH_SIZE, STEP_BETWEEN_STATE, TRAINING_FREQUENCY, TARGET_UPDATE_FREQUENCY, MIN_FRAME_FOR_NEG_REWARD, memory_usage, time_elapsed])
        if epochs % 10 == 0:
            writer.writerow([TEST_NAME, epochs, np.max(reward), agent.epsilon , agent.epsilon_min, agent.epsilon_decay, agent.gamma , agent.memory_size, BATCH_SIZE, STEP_BETWEEN_STATE, TRAINING_FREQUENCY, TARGET_UPDATE_FREQUENCY, MIN_FRAME_FOR_NEG_REWARD, memory_usage, time_elapsed])


def transform_to_grey_scale(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = obs.astype(float)
    obs /= 255.0
    return obs

def memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss

if __name__ == "__main__":
    env = gym.make("CarRacing-v2", max_episode_steps=500, domain_randomize=False)
    
    if LOADING_EXISTING_MODEL:
        agent = DQNAgent(epsilon=0.15)
        agent.load_model(LOADING_MODEL_PATH)
    else:
        agent = DQNAgent()
    


    last_10_rewards = deque(maxlen=10)
    
    start_time = time.time()
    for epoch in tqdm(range(STARTING_EPOCH,NUM_EPOCS)):
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
                reward += r
                if done or info:
                    break
            
            if reward < 0 and frame_counter > MIN_FRAME_FOR_NEG_REWARD:
                negative_reward_counter += 1
            else:
                negative_reward_counter = 0                
                
            # TODO: add a penalty for not moving and make it stop if the reward is too low
            # handle the reward part 
            total_reward += reward
            print(total_reward)
            if(total_reward < 0 or negative_reward_counter > MAX_CONSECUTIVE_NEG_REWARD):
                break
            
            next_frame = transform_to_grey_scale(next_frame)
            agent.save_transition(current_frame, action, reward, next_frame, done) # save the transition in the memory
            saved_transitions += 1

            
            current_frame = next_frame # update the current frame
            frame_counter += STEP_BETWEEN_STATE
            
            if len(agent.memory) > BATCH_SIZE and saved_transitions % TRAINING_FREQUENCY == 0:
                agent.replay(BATCH_SIZE) # train the model with the saved transitions
                saved_transitions = 0
        
        last_10_rewards.append(total_reward)    
        
        if epoch % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_model()
        
        if epoch % MODEL_SAVE_FREQUENCY == 0:
            agent.save_model(f"./save/{TEST_NAME}/model-{epoch}.h5")
        
        memory_usage_string = f"{memory_usage() / (1024 ** 2):.2f} MB"
        time_elapsed = time.time() - start_time
        save_log(epoch, last_10_rewards, agent, memory_usage_string, round(time_elapsed, 1))
                            
           
    env.close()
