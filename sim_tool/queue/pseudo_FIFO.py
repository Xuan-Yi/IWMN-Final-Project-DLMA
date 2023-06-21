import numpy as np
from typing import Tuple


class pseudo_FIFO:
    def __init__(self, n_nodes: int, size: float | Tuple[float], x: float | Tuple[float], mode='CBR'):
        # x: bit arrival rate
        # arrival_mode = 'CBR' (constant bit arrival) or 'PPA' (Poisson pacxet arrival)
        assert np.all((x >= 0) & (x < 1))
        assert mode == 'CBR' or mode == 'PPA'
        self.n_nodes = n_nodes
        self.mode = mode

        self.size = np.array(size, dtype=np.float32)
        self.x = np.array(x, dtype=np.float32)

        self.queue = np.zeros(n_nodes, dtype=np.float32)
        self.data = np.zeros(n_nodes, dtype=np.float32)
        self.loss = np.zeros(n_nodes, dtype=np.float32)

        self.rng = rng = np.random.default_rng()  # for PPA

    def enque(self):
        if self.mode == 'CBR':  # CBR
            arrival = self.x * np.ones(self.n_nodes)
        else:  # PPA
            k = 20/self.x
            arrival = self.rng.poisson(self.x*k, self.n_nodes)/k

        self.data += 1-self.isEmpty().astype(np.float32)
        self.queue += arrival
        loss = np.maximum(0, self.queue-self.size)
        self.loss += loss
        self.queue -= loss

    def deque(self, pick=bool | Tuple[bool]):
        pop = np.minimum(1, self.queue, dtype=np.float32)
        idx = np.where(np.logical_not(pick))
        pop[idx] = 0
        self.queue -= pop

        return pop

    def isEmpty(self):
        return self.queue <= 0
