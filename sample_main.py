from sim_tools.configs.basic_config import BasicConfig
from sim_tools.environments.basic_env import BasicEnvironment
from sim_tools.plot_funcs.plot_throughput import PlotThroughput

import numpy as np

import os

import matplotlib.pyplot as plt
from tqdm import tqdm

EXPERIMENT_NAME = '1 agent + 1 EB-ALOHA'

# configs
config = BasicConfig()
config.n_DQN = 1
config.n_TDMA = 0
config.n_EB_ALOHA = 1
config.n_q_ALOHA = 0

config.max_iter = 10000
config.N = 1000
config.alpha = 1  # default 0

# environment
env = BasicEnvironment(config)

agent_reward_list = []
tdma_reward_list = []
eb_aloha_reward_list = []
q_aloha_reward_list = []

# load pre-trained models
if os.path.isdir(f'./models_{EXPERIMENT_NAME}'):
    env.load_models(f'./models_{EXPERIMENT_NAME}/DQN')

# simulation
for i in tqdm(range(config.max_iter)):
    dqn_rewards, tdma_rewards, eb_aloha_rewards, q_aloha_rewards = env.step()

    agent_reward_list.append(dqn_rewards)
    tdma_reward_list.append(tdma_rewards)
    eb_aloha_reward_list.append(eb_aloha_rewards)
    q_aloha_reward_list.append(q_aloha_rewards)

# save models
if config.save_model:
    env.save_models(f'./models_{EXPERIMENT_NAME}/DQN')
    
# save the results
agent_arr = np.array(agent_reward_list, dtype=np.float32)
tdma_arr = np.array(tdma_reward_list, dtype=np.float32)
eb_ALOHA_arr = np.array(eb_aloha_reward_list, dtype=np.float32)
q_ALOHA_arr = np.array(q_aloha_reward_list, dtype=np.float32)

M, E, F, B, X, W, q, alpha = config.M, config.E, config.F, config.B, config.X, config.W, config.q, config.alpha
n_DQN, n_TDMA, n_EB_ALOHA, n_q_ALOHA = config.n_DQN, config.n_TDMA, config.n_EB_ALOHA, config.n_q_ALOHA
max_iter = config.max_iter

if not os.path.isdir('./rewards'):
    os.mkdir('./rewards')
file_path = f'rewards/{EXPERIMENT_NAME}_alpha{alpha}_dqn{n_DQN}_t{n_TDMA}_ea{n_EB_ALOHA}_qa{n_q_ALOHA}_M{M}_E{E:.0E}_F{F}_B{B}_X{X}_W{W}_q{q}_{max_iter:.0E}.npz'
np.savez(file_path, agent=agent_arr, tdma=tdma_arr,
         eb_ALOHA=eb_ALOHA_arr, q_ALOHA=q_ALOHA_arr)

# plot throughput
fig1 = plt.figure()
PlotThroughput(file_path, config)

plt.show()
