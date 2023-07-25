import numpy as np
import matplotlib.pyplot as plt


def main():
    settings = {
        'PointPillars': [6, 5, 4, 3, 2, 1, 0],
        'SECOND':       [0, 1, 2, 3, 4, 5, 6],
    }
    m_ap = np.array([76.72, 77.74, 76.46, 75.8, 74.63, 75.71, 71.94])
    x = np.array(settings['PointPillars'])
    width = 0.35
    multiplier = 0
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    for agent_type, agent_number in settings.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, agent_number, width, label=agent_type, color='b' if agent_type == 'PointPillars' else 'g')
        # ax1.bar_label(rects, padding=3)
        multiplier += 1

    ax1.set_ylabel('Num Agents', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.set_xticks([])
    ax1.legend(loc='upper left', ncols=2)

    ax2.plot(x, m_ap, 'r-', marker='o', label='Our')
    ax2.set_ylim([55, 80])
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax2.set_ylabel('mAP', fontsize=12)
    ax2.hlines(y=78.1, xmin=0, xmax=len(x)-1, colors='darkorange', label='Early', linestyles='dashed')
    ax2.hlines(y=67.8, xmin=0, xmax=len(x)-1, colors='k', label='Late', linestyles='dashed')
    ax2.legend(loc='upper right', ncols=3)
    plt.show()


if __name__ == '__main__':
    main()
