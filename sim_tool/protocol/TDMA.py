import numpy as np


class TDMA_NODES:
    def __init__(self, n_nodes, action_list_len, X, queue=None):
        # n_actions=2: (wait, transmit)
        # action_list_len and X indicate the parameters of ONE node.
        self.n_nodes = n_nodes
        self.action_list_len = action_list_len
        self.X = X
        self.action_list = self.__create_action_list__()
        self.counter = 0

        self.use_queue = queue is not None  # whether queue should be considered
        self.queue = queue

    def __create_action_list__(self):  # (node, action_list)
        action_list = np.zeros(
            (self.n_nodes, self.action_list_len), dtype=np.float32)
        for i in range(self.n_nodes):
            idx = np.random.choice(self.action_list_len, self.X, replace=False)
            action_list[i, idx] = 1
        return action_list

    def tic(self):  # 1D: action of each node
        tdma_action = self.action_list[:, self.counter]
        # tdma_action = np.squeeze(tdma_action)
        self.counter += 1
        if self.counter == self.action_list.shape[1]:
            self.counter = 0

        throuput = tdma_action.astype(np.float32)
        if self.use_queue:
            self.queue.enque()
            throuput = self.queue.deque(pick=tdma_action.astype(bool))
        return throuput

    def shuffle(self, _X):  # Change the action pattern.
        self.X = _X
        self.__create_action_list__()

    def reset(self):
        self.action_list = self.__create_action_list__()
        self.counter = 0
