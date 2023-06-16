from sim_tools.protocols.DQN import DQN_NODES
from sim_tools.protocols.TDMA import TDMA_NODES
from sim_tools.protocols.EB_Aloha import EB_ALOHA_NODES
from sim_tools.protocols.q_Aloha import q_Aloha_NODES

import numpy as np

import os
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Input, Add
from keras.optimizers import RMSprop, Adam
from keras.initializers import glorot_normal

import matplotlib.pyplot as plt
from tqdm import tqdm


class BasicEnvironment:
    def __init__(self, config):
        self.__set_env__(config)

    def __set_env__(self, _config):
        self.n_DQN = _config.n_DQN
        self.n_TDMA = _config.n_TDMA
        self.n_EB_Aloha = _config. n_EB_Aloha
        self.n_q_Aloha = _config.n_q_Aloha

        self.n_nodes = self.n_DQN + self.n_TDMA + self.n_EB_Aloha + self.n_q_Aloha

        self.dqn_nodes = DQN_NODES(_config.state_size,
                                   n_dqn_nodes=self.n_DQN,
                                   n_nodes=self.n_nodes,
                                   n_actions=2,
                                   memory_size=_config.E,
                                   replace_target_iter=_config.F,
                                   batch_size=_config.B,
                                   learning_rate=0.01,
                                   gamma=0.9,
                                   epsilon=0.5,
                                   epsilon_min=0.005,
                                   epsilon_decay=0.995,
                                   alpha=_config.alpha
                                   )
        self.tdma_nodes = TDMA_NODES(
            _config.n_TDMA, _config.action_list_len, _config.X)
        self.EB_ALOHA_NODES = EB_ALOHA_NODES(
            _config.n_EB_Aloha, _config.W, _config.max_count)
        self.q_Aloha_nodes = q_Aloha_NODES(_config.n_q_Aloha, _config.q)

    def reset(self, _config):
        self.config = _config
        self.__set_env__(self.config)

        self.dqn_nodes.reset()
        self.tdma_nodes.reset()
        self.EB_ALOHA_NODES.reset()
        self.q_Aloha_nodes.reset()

    def step(self):
        dqn_rewards = np.zeros(self.n_DQN)
        tdma_rewards = np.zeros(self.n_TDMA)
        eb_Aloha_rewards = np.zeros(self.n_EB_Aloha)
        q_Aloha_rewards = np.zeros(self.n_q_Aloha)

        dqn_actions = np.zeros(self.n_DQN, dtype=np.float32)
        tdma_actions = np.zeros(self.n_TDMA, dtype=np.float32)
        eb_Aloha_actions = np.zeros(self.n_EB_Aloha, dtype=np.float32)
        q_Aloha_actions = np.zeros(self.n_q_Aloha, dtype=np.float32)

        observation_ = np.array(['I']*self.n_DQN)  # obersvation for DQN nodes

        if self.n_DQN > 0:
            dqn_actions = self.dqn_nodes.tic()
        if self.n_TDMA > 0:
            tdma_actions = self.tdma_nodes.tic()
        if self.n_EB_Aloha > 0:
            eb_Aloha_actions = self.EB_ALOHA_NODES.tic()
        if self.n_q_Aloha > 0:
            q_Aloha_actions = self.q_Aloha_nodes.tic()

        # evaluate media condition
        n_Tx = np.sum(dqn_actions)+np.sum(tdma_actions) + \
            np.sum(eb_Aloha_actions)+np.sum(q_Aloha_actions)
        assert n_Tx >= 0

        if n_Tx == 0:  # idle (default)
            pass
        elif n_Tx == 1:  # success Tx
            dqn_rewards = dqn_actions
            tdma_rewards = tdma_actions
            eb_Aloha_rewards = eb_Aloha_actions
            q_Aloha_rewards = q_Aloha_actions

            self.EB_ALOHA_NODES.handle_success()

            for i in range(self.n_DQN):
                observation_[i] = 'S' if dqn_actions[i] == 1 else 'B'
        else:  # collision
            self.EB_ALOHA_NODES.handle_collision()

            if np.sum(dqn_actions) > 0:
                for i in range(self.n_DQN):
                    observation_[i] = 'F' if dqn_actions[i] == 1 else 'B'
            else:
                observation_ = np.array(['B']*self.n_DQN)

        # update DQN nodes
        non_agent_rewards = np.zeros(
            (self.n_DQN, self.n_nodes-1), dtype=np.float32)
        cat_rewards = np.concatenate(
            (dqn_rewards, tdma_rewards, eb_Aloha_rewards, q_Aloha_rewards), dtype=np.float32)

        for i in range(self.n_DQN):
            non_agent_rewards[i, :i] = cat_rewards[np.newaxis, :i]
            non_agent_rewards[i, i:] = cat_rewards[np.newaxis, i+1:]

        self.dqn_nodes.update(observation_, dqn_rewards, non_agent_rewards)

        return dqn_rewards, tdma_rewards, eb_Aloha_rewards, q_Aloha_rewards
