from sim_tools.protocols.DQN import DQN_NODES
from sim_tools.protocols.TDMA import TDMA_NODES
from sim_tools.protocols.EB_ALOHA import EB_ALOHA_NODES
from sim_tools.protocols.q_ALOHA import q_ALOHA_NODES

import numpy as np


class BasicEnvironment:
    def __init__(self, config):
        self.__set_env__(config)

    def __set_env__(self, _config):
        # Cannot change the number DQN nodes
        if not hasattr(self, 'n_DQN'):
            self.n_DQN = _config.n_DQN
        self.n_TDMA = _config.n_TDMA
        self.n_EB_ALOHA = _config. n_EB_ALOHA
        self.n_q_ALOHA = _config.n_q_ALOHA

        self.n_nodes = self.n_DQN + self.n_TDMA + self.n_EB_ALOHA + self.n_q_ALOHA

        if not hasattr(self, 'dqn_nodes'):
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
        else:
            self.dqn_nodes.n_nodes = self.n_nodes
        self.tdma_nodes = TDMA_NODES(
            _config.n_TDMA, _config.action_list_len, _config.X)
        self.EB_ALOHA_NODES = EB_ALOHA_NODES(
            _config.n_EB_ALOHA, _config.W, _config.max_count)
        self.q_ALOHA_nodes = q_ALOHA_NODES(_config.n_q_ALOHA, _config.q)

    def reset(self, _config):
        self.config = _config
        self.__set_env__(self.config)

        # self.dqn_nodes.reset()
        self.tdma_nodes.reset()
        self.EB_ALOHA_NODES.reset()
        self.q_ALOHA_nodes.reset()

    def step(self):
        dqn_rewards = np.zeros(self.n_DQN)
        tdma_rewards = np.zeros(self.n_TDMA)
        eb_aloha_rewards = np.zeros(self.n_EB_ALOHA)
        q_aloha_rewards = np.zeros(self.n_q_ALOHA)

        dqn_actions = np.zeros(self.n_DQN, dtype=np.float32)
        tdma_actions = np.zeros(self.n_TDMA, dtype=np.float32)
        eb_aloha_actions = np.zeros(self.n_EB_ALOHA, dtype=np.float32)
        q_aloha_actions = np.zeros(self.n_q_ALOHA, dtype=np.float32)

        observation_ = np.array(['I']*self.n_DQN)  # obersvation for DQN nodes

        if self.n_DQN > 0:
            dqn_actions = self.dqn_nodes.tic()
        if self.n_TDMA > 0:
            tdma_actions = self.tdma_nodes.tic()
        if self.n_EB_ALOHA > 0:
            eb_aloha_actions = self.EB_ALOHA_NODES.tic()
        if self.n_q_ALOHA > 0:
            q_aloha_actions = self.q_ALOHA_nodes.tic()

        # evaluate media condition
        n_Tx = np.sum(dqn_actions)+np.sum(tdma_actions) + \
            np.sum(eb_aloha_actions)+np.sum(q_aloha_actions)
        assert n_Tx >= 0

        if n_Tx == 0:  # idle (default)
            pass
        elif n_Tx == 1:  # success Tx
            dqn_rewards = dqn_actions
            tdma_rewards = tdma_actions
            eb_aloha_rewards = eb_aloha_actions
            q_aloha_rewards = q_aloha_actions

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
            (dqn_rewards, tdma_rewards, eb_aloha_rewards, q_aloha_rewards), dtype=np.float32)

        for i in range(self.n_DQN):
            non_agent_rewards[i, :i] = cat_rewards[np.newaxis, :i]
            non_agent_rewards[i, i:] = cat_rewards[np.newaxis, i+1:]

        self.dqn_nodes.update(observation_, dqn_rewards, non_agent_rewards)

        return dqn_rewards, tdma_rewards, eb_aloha_rewards, q_aloha_rewards
