import numpy as np
import random


class q_ALOHA_NODES:
    def __init__(self, n_nodes: int, q=0.5, queue=None):
        # n_actions=2: (wait, transmit)
        # queue: queue from 'sim_tool.queue' or None (don't consider queue)
        self.n_nodes = n_nodes
        self.q = np.array(q)  # probability to send
        assert (np.all((self.q <= 1) & (self.q >= 0)))
        self.actions = np.zeros(self.n_nodes, dtype=np.float32)

        self.use_queue = queue is not None  # whether queue should be considered
        self.queue = queue

    def tic(self):
        # return 1 with prob. q
        if self.q.size == 1:
            action_list = np.random.choice(
                2, self.n_nodes, p=[1-self.q, self.q])
        else:
            action_list = np.zeros(self.n_nodes)
            for i, _q in enumerate(self.q):
                action_list[i] = random.choices([0, 1], weights=[1-_q, _q])[0]
        
        throuput = action_list
        if self.use_queue:
            self.queue.enque()
            throuput = self.queue.deque(pick=action_list.astype(bool))
        return throuput

    def reset(self):  # Change the action pattern.
        pass
