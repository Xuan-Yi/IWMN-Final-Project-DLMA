class BasicConfig:
    def __init__(self):
        self.n_DQN = 1
        self.n_TDMA = 0
        self.n_EB_ALOHA = 0
        self.n_q_ALOHA = 0

        self.max_iter = 10000  # simulation iterations
        self.N = 1000  # plot with avg of N iters

        # Agent (DQN)
        self.M = 20  # state length
        self.E = 500  # memory size
        self.F = 20  # target network update frequency
        self.B = 32  # mini-batch size
        self.alpha = 0  # alpha-fairness
        # state = cat(s[8:], [action, observation, agent_reward, non_agent_reward])
        self.state_size = int(8*self.M)

        # TDMA
        self.action_list_len = 10  # length of one period
        self.X = 2  # number of slot used in one perios

        # Exponential-backoff ALOHA
        # wnd = randint(0, W*2^count)
        self.W = 2   # minimum window size
        self.max_count = 2  # maximum backoff count

        # q-ALOHA
        self.q = .2
        

