import numpy as np
import matplotlib.pyplot as plt


def main():
    num_agents = np.arange(0, 6)
    perf = np.array([52.84, 67.93, 72.47, 75.82, 77.35, 77.74])
    fig, ax = plt.subplots()
    ax.plot(num_agents, perf, color='r', marker='o', label='Our')
    ax.set_xticks(num_agents)
    ax.set_xlabel('num connected agents', fontsize=12)
    ax.set_ylabel('AP', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.hlines(y=78.1, xmin=0, xmax=num_agents[-1] + 0.15, colors='darkorange', label='Early', linestyles='dashed')
    ax.legend(loc='upper left', ncols=1)
    ax.grid()
    plt.show()


if __name__ == '__main__':
    main()

