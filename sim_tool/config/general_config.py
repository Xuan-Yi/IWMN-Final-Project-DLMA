class GeneralConfig:
    def __init__(self):
        self.n_DQN = 1
        self.n_TDMA = 0
        self.n_EB_ALOHA = 0
        self.n_q_ALOHA = 0

        self.max_iter = 10000  # simulation iterations
        self.N = 1000  # plot with avg of N iters

        self.save_model = True

        # Agent (DQN)
        self.M = 20  # state length
        self.E = 500  # memory size
        self.F = 20  # target network update frequency
        self.B = 32  # mini-batch size
        self.alpha = 0  # alpha-fairness
        # state = [action, observation, agent_reward, non_agent_reward]
        self.state_size = int(8*self.M)

        self.DQN_size = 1000  # queue size
        self.DQN_x = .2  # queue x
        self.DQN_mode = 'PPA'

        # TDMA
        self.action_list_len = 10  # length of one period
        self.X = 2  # number of slot used in one perios

        self.TDMA_size = 1000  # queue size
        self.TDMA_x = .2  # queue x
        self.TDMA_mode = 'PPA'

        # Exponential-backoff ALOHA
        # wnd = randint(0, W*2^count)
        self.W = 2   # minimum window size
        self.max_count = 2  # maximum backoff count

        self.EB_ALOHA_size = 1000  # queue size
        self.EB_ALOHA_x = .2  # queue x
        self.EB_ALOHA_mode = 'PPA'

        # q-ALOHA
        self.q = .2

        self.q_ALOHA_size = 1000  # queue size
        self.q_ALOHA_x = .2  # queue x
        self.q_ALOHA_mode = 'PPA'
