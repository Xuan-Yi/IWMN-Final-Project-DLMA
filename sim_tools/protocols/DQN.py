import numpy as np

import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Input, Add
from keras.optimizers import Adam

class DQN_NODES:
    def __init__(self,
                 state_size,
                 n_dqn_nodes,  # K: number of DQN nodes
                 n_nodes,  # N: number of all nodes
                 n_actions,
                 memory_size=500,
                 replace_target_iter=200,
                 batch_size=32,
                 learning_rate=0.01,
                 gamma=0.9,
                 epsilon=1,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 alpha=0  # 0 ~ 100 (inf)
                 ):
        # hyper-parameters
        self.state_size = state_size
        self.n_dqn_nodes = n_dqn_nodes
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha

        self.reset()

    def __create_agents__(self):
        self.agents = []

        for i in range(self.n_dqn_nodes):
            dqn_agent = DQN(self.state_size,
                            n_nodes=self.n_nodes,
                            n_actions=self.n_actions,
                            memory_size=self.memory_size,
                            replace_target_iter=self.replace_target_iter,
                            batch_size=self.batch_size,
                            learning_rate=self.learning_rate,
                            gamma=self.gamma,
                            epsilon=self.epsilon,
                            epsilon_min=self.epsilon_min,
                            epsilon_decay=self.epsilon_decay,
                            alpha=self.alpha
                            )
            self.agents.append(dqn_agent)

    def reset(self):
        self.__create_agents__()

    def tic(self):
        agent_actions = np.zeros(self.n_dqn_nodes, dtype=np.float32)

        for i in range(self.n_dqn_nodes):
            agent_actions[i] = self.agents[i].tic()
        return agent_actions

    def update(self, observations_, agent_rewards, non_agent_rewards):
        for i in range(self.n_dqn_nodes):
            # print(observations_[i], agent_rewards[i], non_agent_rewards[i])
            self.agents[i].update(
                observations_[i], agent_rewards[i], non_agent_rewards[i])

class DQN:
    def __init__(self,
                 state_size,
                 n_nodes,  # N: number of all nodes
                 n_actions,
                 memory_size=500,
                 replace_target_iter=200,
                 batch_size=32,
                 learning_rate=0.01,
                 gamma=0.9,
                 epsilon=1,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 alpha=0  # 0 ~ 100 (inf)
                 ):
        # hyper-parameters
        self.state_size = state_size
        self.n_nodes = n_nodes
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha

        self.reset()

    def reset(self):
        self.state = np.zeros(self.state_size)  # init state
        self.n_iter = 0  # current iteration

        # [s, a, r1, r2, ..., s_]
        self.memory = np.zeros(
            shape=(self.memory_size, self.state_size * 2 + (self.n_nodes + 1)))
        # temporary parameters
        self.learn_step_counter = 0
        self.memory_couter = 0

        # build model
        self.model = self.__build_ResNet_model__()  # model: evaluate Q value
        self.target_model = self.__build_ResNet_model__()  # target_mode: target network

    def tic(self):
        self.n_iter += 1
        self.agent_action = np.array(
            self.__choose_action__(self.state), dtype=np.float32)

        return self.agent_action

    def update(self, observation_, agent_reward, non_agent_reward):
        # non_agent_reward: 1D array or scalar
        next_state = np.concatenate((self.state[8:], np.array(self.__return_action__(
            self.agent_action) + self.__return_observation__(observation_) + [agent_reward, np.sum(non_agent_reward, dtype=np.float32)], dtype=np.float32)))

        self.__store_transition__(
            self.state, self.agent_action, agent_reward, non_agent_reward, next_state)

        if self.n_iter > 100:
            self.__learn__()    # internally iterates default (prediction) model

        self.state = next_state

    def __return_action__(self, action):
        one_hot_vector = [0] * self.n_actions
        one_hot_vector[int(action)] = 1
        return one_hot_vector

    def __return_observation__(self, o):
        if o == 'S':
            return [1, 0, 0, 0]
        elif o == 'F':
            return [0, 1, 0, 0]
        elif o == 'B':
            return [0, 0, 1, 0]
        elif o == 'I':
            return [0, 0, 0, 1]
        else:
            print(f'error obervation: {o}')

    def __alpha_function__(self, action_values):
        if self.alpha == 1:
            log_action_values = np.log(action_values, dtype=np.float32)
            action_values_list = [np.sum(log_action_values[self.n_nodes*j: self.n_nodes*(
                j+1)], dtype=np.float32) for j in range(self.n_actions)]
        elif self.alpha == 0:
            action_values_list = [np.sum(action_values[self.n_nodes*j: self.n_nodes*(
                j+1)], dtype=np.float32) for j in range(self.n_actions)]
        elif self.alpha == 100:
            action_values_list = [
                np.amin(action_values[self.n_nodes*j: self.n_nodes*(j+1)], axis=0) for j in range(self.n_actions)]
        else:
            pow_action_values = np.power(
                action_values, (1-self.alpha), dtype=np.float32)
            action_values_list = [1/(1-self.alpha) * np.sum(pow_action_values[self.n_nodes *
                                                                              j: self.n_nodes*(j+1)], dtype=np.float32) for j in range(self.n_actions)]

        return np.argmax(action_values_list)

    def __build_ResNet_model__(self):
        inputs = Input(shape=(self.state_size, ))
        h1 = Dense(64, activation="relu",
                   kernel_initializer='glorot_normal')(inputs)  # h1
        h2 = Dense(64, activation="relu",
                   kernel_initializer='glorot_normal')(h1)  # h2

        h3 = Dense(64, activation="relu",
                   kernel_initializer='glorot_normal')(h2)  # h3
        h4 = Dense(64, activation="relu",
                   kernel_initializer='glorot_normal')(h3)  # h4
        add1 = Add()([h4, h2])

        h5 = Dense(64, activation="relu",
                   kernel_initializer='glorot_normal')(add1)  # h5
        h6 = Dense(64, activation="relu",
                   kernel_initializer='glorot_normal')(h5)  # h6
        add2 = Add()([h6, add1])

        outputs = Dense(
            self.n_actions*self.n_nodes, kernel_initializer='glorot_normal')(add2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=Adam(
            learning_rate=self.learning_rate))
        return model

    def __choose_action__(self, state):
        # Apply epsilon-greedy algorithm
        state = state[np.newaxis, :]
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)

        action_values = self.model.predict(state, verbose=None)
        return self.__alpha_function__(action_values[0])

    def __store_transition__(self, s, a, r_dqn, r_non_dqn, s_):
        # s_: next_state
        if not hasattr(self, 'memory_couter'):
            self.memory_couter = 0
        transition = np.concatenate((s, [a, r_dqn], r_non_dqn, s_))
        index = self.memory_couter % self.memory_size
        self.memory[index, :] = transition
        self.memory_couter += 1

    def __repalce_target_parameters__(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def __learn__(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.__repalce_target_parameters__()  # iterative target model
        self.learn_step_counter += 1

        if self.memory_couter > self.memory_size:
            sample_index = np.random.choice(
                self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(
                self.memory_couter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # batch memory row: [s, a, r1, r2, ..., s_]
        state = batch_memory[:, :self.state_size]
        action = batch_memory[:, self.state_size].astype(
            np.int32)  # float -> int
        rewards = batch_memory[:, self.state_size +
                               1: self.state_size+self.n_nodes+1]  # [:, (r1, r2, ...)]
        next_state = batch_memory[:, -self.state_size:]

        q = self.model.predict(state, verbose=None)  # state
        q_targ = self.target_model.predict(
            next_state, verbose=None)  # next state

        for i in range(self.batch_size):
            action_ = self.__alpha_function__(q_targ[i])

            # action_:
            # |      a0      |      a1      |
            # | 01 | 02 | 03 | 01 | 02 | 03 |
            for node in range(self.n_nodes):
                q[i, self.n_nodes*action[i]+node] = rewards[i, node] + \
                    self.gamma*q_targ[i][self.n_nodes*action_+node]

        self.model.fit(state, q, self.batch_size, epochs=1, verbose=None)