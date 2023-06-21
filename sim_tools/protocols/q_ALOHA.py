import numpy as np
import random


class q_ALOHA_NODES:
    def __init__(self, n_nodes, q=0.5):
        # n_actions=2: (wait, transmit)
        self.n_nodes = n_nodes
        self.q = np.array(q)  # probability to send
        assert (np.all((self.q <= 1) & (self.q >= 0)))
        self.actions = np.zeros(self.n_nodes, dtype=np.float32)

    def tic(self):
        # return 1 with prob. q
        if self.q.size == 1:
            action_list = np.random.choice(
                2, self.n_nodes, p=[1-self.q, self.q])
        else:
            action_list = np.zeros(self.n_nodes)
            for i, _q in enumerate(self.q):
                action_list[i] = random.choices([0, 1], weights=[1-_q, _q])[0]
        return action_list

    def reset(self):  # Change the action pattern.
        pass
