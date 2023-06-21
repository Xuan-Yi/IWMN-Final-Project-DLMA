from sim_tool.protocol.DQN import DQN_NODES
from sim_tool.protocol.TDMA import TDMA_NODES
from sim_tool.protocol.EB_ALOHA import EB_ALOHA_NODES
from sim_tool.protocol.q_ALOHA import q_ALOHA_NODES

from sim_tool.queue.pseudo_FIFO import pseudo_FIFO

import numpy as np


class GeneralEnvironment:
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

        self.DQN_queue = pseudo_FIFO(
            self.n_DQN, _config.DQN_size, _config.DQN_x, _config.DQN_mode)
        self.TDMA_queue = pseudo_FIFO(
            self.n_TDMA, _config.TDMA_size, _config.TDMA_x, _config.TDMA_mode)
        self.EB_ALOHA_queue = pseudo_FIFO(
            self.n_EB_ALOHA, _config.EB_ALOHA_size, _config.EB_ALOHA_x, _config.EB_ALOHA_mode)
        self.q_ALOHA_queue = pseudo_FIFO(
            self.n_q_ALOHA, _config.q_ALOHA_size, _config.q_ALOHA_x, _config.q_ALOHA_mode)

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
                                       alpha=_config.alpha,
                                       queue=self.DQN_queue
                                       )
        else:
            self.dqn_nodes.n_nodes = self.n_nodes
        self.tdma_nodes = TDMA_NODES(
            _config.n_TDMA, _config.action_list_len, _config.X, queue=self.TDMA_queue)
        self.EB_ALOHA_NODES = EB_ALOHA_NODES(
            _config.n_EB_ALOHA, _config.W, _config.max_count, queue=self.EB_ALOHA_queue)
        self.q_ALOHA_nodes = q_ALOHA_NODES(
            _config.n_q_ALOHA, _config.q, queue=self.q_ALOHA_queue)

    def save_models(self, filename):
        self.dqn_nodes.save(filename)

    def load_models(self, filename):
        self.dqn_nodes.load(filename)

    def reset(self, _config):
        self.config = _config
        self.__set_env__(self.config)

        # self.dqn_nodes.reset()
        self.tdma_nodes.reset()
        self.EB_ALOHA_NODES.reset()
        self.q_ALOHA_nodes.reset()

    def getRxData(self):
        return self.DQN_queue.data, self.TDMA_queue.data, self.EB_ALOHA_queue.data, self.q_ALOHA_queue.data

    def getRxLoss(self):
        return self.DQN_queue.loss, self.TDMA_queue.loss, self.EB_ALOHA_queue.loss, self.q_ALOHA_queue.loss

    def step(self):
        dqn_rewards = np.zeros(self.n_DQN)
        tdma_rewards = np.zeros(self.n_TDMA)
        eb_aloha_rewards = np.zeros(self.n_EB_ALOHA)
        q_aloha_rewards = np.zeros(self.n_q_ALOHA)

        dqn_throughput = np.zeros(self.n_DQN, dtype=np.float32)
        tdma_throughput = np.zeros(self.n_TDMA, dtype=np.float32)
        eb_aloha_throughput = np.zeros(self.n_EB_ALOHA, dtype=np.float32)
        q_aloha_throughput = np.zeros(self.n_q_ALOHA, dtype=np.float32)

        observation_ = np.array(['I']*self.n_DQN)  # obersvation for DQN nodes

        if self.n_DQN > 0:
            dqn_throughput = self.dqn_nodes.tic()
        if self.n_TDMA > 0:
            tdma_throughput = self.tdma_nodes.tic()
        if self.n_EB_ALOHA > 0:
            eb_aloha_throughput = self.EB_ALOHA_NODES.tic()
        if self.n_q_ALOHA > 0:
            q_aloha_throughput = self.q_ALOHA_nodes.tic()

        # evaluate media condition
        n_Tx = np.count_nonzero(dqn_throughput)+np.count_nonzero(tdma_throughput) + \
            np.count_nonzero(eb_aloha_throughput) + \
            np.count_nonzero(q_aloha_throughput)
        assert n_Tx >= 0

        if n_Tx == 0:  # idle (default)
            pass
        elif n_Tx == 1:  # success Tx
            dqn_rewards = dqn_throughput
            tdma_rewards = tdma_throughput
            eb_aloha_rewards = eb_aloha_throughput
            q_aloha_rewards = q_aloha_throughput

            self.EB_ALOHA_NODES.handle_success()

            for i in range(self.n_DQN):
                observation_[i] = 'S' if dqn_throughput[i] > 0 else 'B'
        else:  # collision
            self.EB_ALOHA_NODES.handle_collision()

            if np.sum(dqn_throughput) > 0:
                for i in range(self.n_DQN):
                    observation_[i] = 'F' if dqn_throughput[i] > 0 else 'B'
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
