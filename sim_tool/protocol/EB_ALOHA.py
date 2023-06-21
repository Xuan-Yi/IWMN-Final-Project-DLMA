import numpy as np


class EB_ALOHA_NODES:
    def __init__(self, n_nodes, W=2, max_count=2, queue=None):
        # n_actions=2: (wait, transmit)
        self.n_nodes = n_nodes
        self.max_count = max_count
        self.W = W
        self.actions = np.zeros(self.n_nodes, dtype=np.float32)

        self.count = np.zeros(self.n_nodes, dtype=np.float32)
        self.backoff = np.random.randint(
            0, self.W * 2**self.count, size=self.n_nodes)

        self.use_queue = queue is not None  # whether queue should be considered
        self.queue = queue

    def tic(self):
        self.count = np.minimum(self.count, self.max_count)
        self.backoff -= 1

        filter_arr = self.backoff < 0
        filter_arr = np.arange(self.n_nodes, dtype=np.int32)[filter_arr]
        self.backoff[filter_arr] = np.random.randint(
            0, self.W * 2**self.count)[filter_arr]

        eb_aloha_actions = (self.backoff == 0)
        eb_aloha_actions = eb_aloha_actions.astype(np.float32)
        self.actions = eb_aloha_actions

        throuput = eb_aloha_actions
        if self.use_queue:
            self.queue.enque()
            throuput = self.queue.deque(pick=eb_aloha_actions.astype(bool))
            idx = np.where(eb_aloha_actions > 0)
            self.actions[idx] = 1

        return throuput  # send if timeout

    def handle_success(self):
        filter_arr = (self.actions == 1)
        self.count[filter_arr] = np.zeros(
            self.n_nodes, dtype=np.int32)[filter_arr]

    def handle_collision(self):
        filter_arr = (self.actions == 1)
        self.count += filter_arr.astype(np.int32)

    def reset(self):  # Change the action pattern.
        self.count = np.zeros(self.n_nodes)
        self.backoff = np.random.randint(
            0, self.W * 2**self.count, size=self.n_nodes)
