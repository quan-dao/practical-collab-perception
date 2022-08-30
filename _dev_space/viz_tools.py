import numpy as np
import torch
from _dev_space.tools_box import apply_tf
import matplotlib.pyplot as plt


def viz_boxes(boxes: np.ndarray):
    xs = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float) / 2.0
    ys = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float) / 2.0
    zs = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float) / 2.0
    out = []
    for i in range(boxes.shape[0]):
        box = boxes[i]
        dx, dy, dz = box[3: 6].tolist()
        vers = np.concatenate([xs.reshape(-1, 1) * dx, ys.reshape(-1, 1) * dy, zs.reshape(-1, 1) * dz], axis=1)  # (8, 3)
        ref_from_box = np.eye(4)
        yaw = box[6]
        cy, sy = np.cos(yaw), np.sin(yaw)
        ref_from_box[:3, :3] = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])
        ref_from_box[:3, 3] = box[:3]
        vers = apply_tf(ref_from_box, vers)
        out.append(vers)
    return out


def print_dict(d: dict):
    for k, v in d.items():
        out = f"{k} | {type(v)} | "
        if isinstance(v, str):
            out += v
        elif isinstance(v, np.ndarray):
            out += f"{v.shape}"
        elif isinstance(v, float):
            out += f"{v}"
        elif isinstance(v, np.bool_):
            out += f"{v.item()}"
        elif isinstance(v, torch.Tensor):
            out += f"{v.shape}"
        print(out)


def viz_clusters2d(xy, labels, ax, marker='o', prefix=''):
    assert len(xy.shape) == 2 and len(labels.shape) == 1
    assert xy.shape[0] == labels.shape[0]
    unq_labels, indices, counts = np.unique(labels, return_inverse=True, return_counts=True)

    # centroid for annotation only
    centroid = np.zeros((unq_labels.shape[0], 2))
    np.add.at(centroid, indices, xy)
    centroid = centroid / counts.reshape(-1, 1)

    colors_palette = np.array([plt.cm.Spectral(each)[:3] for each in np.linspace(0, 1, unq_labels.shape[0])])
    xy_colors = colors_palette[indices]
    xy_colors[labels == -1] = np.zeros(3)  # black for noise

    ax.scatter(xy[:, 0], xy[:, 1], c=xy_colors, marker=marker)
    for cl_idx in range(unq_labels.shape[0]):
        if unq_labels[cl_idx] == -1:
            # noise cluster
            continue
        ax.annotate(f"{prefix}{unq_labels[cl_idx]}", tuple(centroid[cl_idx].tolist()), color='r')


