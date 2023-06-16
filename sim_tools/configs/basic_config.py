class BasicConfig:
    def __init__(self):
        self.n_DQN
        self.n_TDMA
        self.n_EB_Aloha
        self.n_q_Aloha

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

        # Exponential-backoff Aloha
        # wnd = randint(0, W*2^count)
        self.W = 2   # minimum window size
        self.max_count = 2  # maximum backoff count

        # q-Aloha
        self.q = .2
        

