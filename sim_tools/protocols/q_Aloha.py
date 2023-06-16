import numpy as np


class q_Aloha_NODES:
    def __init__(self, n_nodes, q=0.5):
        # n_actions=2: (wait, transmit)
        self.n_nodes = n_nodes
        assert (q <= 1 and q >= 0)
        self.q = q  # probability to send
        self.actions = np.zeros(self.n_nodes, dtype=np.float32)

    def tic(self):
        # return 1 with prob. q
        return np.random.choice(2, self.n_nodes, p=[1-self.q, self.q])

    def reset(self):  # Change the action pattern.
        pass
