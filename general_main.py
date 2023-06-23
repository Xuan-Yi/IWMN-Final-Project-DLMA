from sim_tool.config.general_config import GeneralConfig
from sim_tool.environment.general_env import GeneralEnvironment
from sim_tool.plot_func.plot_throughput import PlotThroughput

import numpy as np

import os

import matplotlib.pyplot as plt
from tqdm import tqdm

EXPERIMENT_NAME = '1 agent + 1 EB-ALOHA'

# configs
config = GeneralConfig()
config.n_DQN = 1
config.n_TDMA = 0
config.n_EB_ALOHA = 1
config.n_q_ALOHA = 0

config.DQN_mode = 'PPA'  # 'PPA' or 'CBR'
config.TDMA_mode = 'PPA'  # 'PPA' or 'CBR'
config.EB_ALOHA_mode = 'PPA'  # 'PPA' or 'CBR'
config.q_ALOHA_mode = 'PPA'   # 'PPA' or 'CBR'

config.max_iter = 10000
config.N = 1000
config.alpha = 1  # default 0

# environment
env = GeneralEnvironment(config)

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

DQN_data, TDMA_data, EB_ALOHA_data, q_ALOHA_data = env.getRxData()
DQN_loss, TDMA_loss, EB_ALOHA_loss, q_ALOHA_loss = env.getRxLoss()

norm_agent_arr = agent_arr*(config.max_iter/DQN_data)
norm_tdma_arr = tdma_arr*(config.max_iter/TDMA_data)
norm_eb_ALOHA_arr = eb_ALOHA_arr*(config.max_iter/EB_ALOHA_data)
norm_q_ALOHA_arr = q_ALOHA_arr*(config.max_iter/q_ALOHA_data)

M, E, F, B, X, W, q, alpha = config.M, config.E, config.F, config.B, config.X, config.W, config.q, config.alpha
n_DQN, n_TDMA, n_EB_ALOHA, n_q_ALOHA = config.n_DQN, config.n_TDMA, config.n_EB_ALOHA, config.n_q_ALOHA
max_iter = config.max_iter

if not os.path.isdir('./rewards'):
    os.mkdir('./rewards')
file_path = f'{EXPERIMENT_NAME}_alpha{alpha}_dqn{n_DQN}_t{n_TDMA}_ea{n_EB_ALOHA}_qa{n_q_ALOHA}_M{M}_E{E:.0E}_F{F}_B{B}_X{X}_W{W}_q{q}_{max_iter:.0E}.npz'
raw_file_path = f'rewards/raw_{file_path}'
normalized_file_path = f'rewards/norm_{file_path}'
np.savez(raw_file_path, agent=agent_arr, tdma=tdma_arr,
         eb_ALOHA=eb_ALOHA_arr, q_ALOHA=q_ALOHA_arr)
np.savez(normalized_file_path, agent=norm_agent_arr, tdma=norm_tdma_arr,
         eb_ALOHA=norm_eb_ALOHA_arr, q_ALOHA=norm_q_ALOHA_arr)

# plot throughput
fig1 = plt.figure('raw')
PlotThroughput(raw_file_path, config)
fig2 = plt.figure('normalized')
PlotThroughput(normalized_file_path, config)

plt.show()
