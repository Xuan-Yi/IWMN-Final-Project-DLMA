import numpy as np
import matplotlib.pyplot as plt


def PlotThroughput(file, config):
    max_iter = config.max_iter
    N = config.N
    n_DQN = config.n_DQN
    n_TDMA = config.n_TDMA
    n_EB_ALOHA = config.n_EB_ALOHA
    n_q_ALOHA = config.n_q_ALOHA

    num = [n_DQN, n_TDMA, n_EB_ALOHA, n_q_ALOHA]
    category = ['agent', 'tdma', 'eb_ALOHA', 'q_ALOHA']

    # load reward
    data = np.load(file)

    labels = []
    rewards = np.zeros((sum(num), max_iter), dtype=np.float32)  # reward

    cnt = 0
    for i in range(len(category)):
        _data = np.transpose(data[category[i]])
        for n in range(num[i]):
            lbl = f'{category[i]} {n+1}' if num[i] > 1 else f'{category[i]}'
            labels.append(lbl)
            rewards[cnt, :] = _data[n]
            cnt += 1

    avg_throughput = np.zeros((sum(num), max_iter), dtype=np.float32)
    temp_sum = np.zeros((sum(num), 1), dtype=np.float32)

    for i in range(0, max_iter):
        if i < N:
            temp_sum[:, 0] += rewards[:, i]
            avg_throughput[:, i] = temp_sum[:, 0]/(i+1)
        else:
            temp_sum[:, 0] += rewards[:, i]-rewards[:, i-N]
            avg_throughput[:, i] = temp_sum[:, 0]/N

    avg_throughput_total = np.sum(avg_throughput, axis=0, dtype=np.float32)

    plt.xlim((0, max_iter))
    plt.ylim((-0.05, 1))

    legend_list = []

    for i in range(len(avg_throughput)):
        line, = plt.plot(avg_throughput[i], lw=1, label=labels[i])
        legend_list.append(line)

    total_line, = plt.plot(avg_throughput_total,
                           color='r', lw=1.5, label='total')
    legend_list.append(total_line)

    plt.grid()
    plt.legend(handles=legend_list, loc='best')
    plt.xlabel("iteration")
    plt.ylabel("average throughput")
