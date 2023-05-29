import numpy as np
import matplotlib.pyplot as plt


def search_rectangle_fitting(points: np.ndarray, delta: float = 0.1):
    assert len(points.shape) == 2
    assert points.shape[1] >= 2
    xy = points[:, : 2]
    queue = list()
    thetas = np.arange(start=0., stop=np.pi/2.0, step=delta)
    for _theta in thetas:
        cos, sin = np.cos(_theta), np.sin(_theta)
        e1 = np.array([cos, sin])
        e2 = np.array([-sin, cos])

        C1 = xy @ e1  # (N,)
        C2 = xy @ e2  # (N,)

        q = criterion_closeness(C1, C2)
        queue.append(q)
    
    queue = np.array(queue)
    theta_star = thetas[np.argmax(queue)]

    cos, sin = np.cos(theta_star), np.sin(theta_star)
    C1_star = xy @ np.array([cos, sin])
    C2_star = xy @ np.array([-sin, cos])

    a1 = cos
    b1 = sin
    c1 = np.min(C1_star)

    a2 = -sin
    b2 = cos
    c2 = np.min(C2_star)

    a3 = cos
    b3 = sin
    c3 = np.max(C1_star)

    a4 = -sin
    b4 = cos
    c4 = np.max(C2_star)

    return np.array([
        [a1, b1, c1],
        [a2, b2, c2],
        [a3, b3, c3],
        [a4, b4, c4]
    ])


def criterion_area(C1: np.ndarray, C2: np.ndarray):
    assert len(C1.shape) == len(C2.shape) == 1
    assert C1.shape == C2.shape
    c1_max, c1_min = C1.max(), C1.min()
    c2_max, c2_min = C2.max(), C2.min()
    cost = -(c1_max - c1_min) * (c2_max - c2_min)
    return cost


def criterion_closeness(C1: np.ndarray, C2: np.ndarray, d0: float = 0.01):
    assert len(C1.shape) == len(C2.shape) == 1
    assert C1.shape == C2.shape
    c1_max, c1_min = C1.max(), C1.min()
    c2_max, c2_min = C2.max(), C2.min()

    c1_max_diff = np.abs(c1_max - C1) # (N,)
    c1_min_diff = np.abs(C1 - c1_min)  # (N,)
    D1 = np.min(np.stack([c1_max_diff, c1_min_diff], axis=1), axis=1)

    c2_max_diff = np.abs(c2_max - C2) # (N,)
    c2_min_diff = np.abs(C2 - c2_min)  # (N,)
    D2 = np.min(np.stack([c2_max_diff, c2_min_diff], axis=1), axis=1)

    cost = 0
    for idx in range(D1.shape[0]):
        d = max([min(D1[idx], D2[idx]), d0])
        cost = cost + 1.0 / d

    return cost

def _calc_closeness_criterion(c1, c2):
    c1_max = np.max(c1)
    c2_max = np.max(c2)
    c1_min = np.min(c1)
    c2_min = np.min(c2)

    D1 = [min([np.linalg.norm(c1_max - ic1),
                np.linalg.norm(ic1 - c1_min)]) for ic1 in c1]
    D2 = [min([np.linalg.norm(c2_max - ic2),
                np.linalg.norm(ic2 - c2_min)]) for ic2 in c2]

    beta = 0
    for i, _ in enumerate(D1):
        d = max(min([D1[i], D2[i]]), 0.01)
        beta += (1.0 / d)

    return beta



def main():
    points = np.load('artifact/hdbscan_dataset10_cluster59.npy')
    rect = search_rectangle_fitting(points)

    rect[:, -1] *= -1

    p01 = np.cross(rect[0], rect[1])
    p12 = np.cross(rect[1], rect[2])
    p23 = np.cross(rect[2], rect[3])
    p30 = np.cross(rect[3], rect[0])
    vers = np.stack([p01, p12, p23, p30], axis=0)
    vers /= vers[:, [-1]]
    print('vers:\n', vers)
    
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1])
    ax.plot(vers[[0, 1], 0], vers[[0, 1], 1], 'r-')
    ax.plot(vers[[1, 2], 0], vers[[1, 2], 1], 'g-')
    ax.plot(vers[[2, 3], 0], vers[[2, 3], 1], 'b-')
    ax.plot(vers[[3, 0], 0], vers[[3, 0], 1], 'k-')
    plt.show()
    print(rect)


if __name__ == '__main__':
    main()

