import numpy as np
import torch
from _dev_space.tools_box import apply_tf
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import colorsys


def viz_boxes(boxes: np.ndarray):
    # box convention:
    # forward: 0 - 1 - 2 - 3, backward: 4 - 5 - 6 - 7, up: 0 - 1 - 5 - 4

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
    print('{')
    for k, v in d.items():
        out = f"\t{k} | {type(v)} | "
        if isinstance(v, str):
            out += v
        elif isinstance(v, np.ndarray):
            out += f"{v.shape}"
        elif isinstance(v, float) or isinstance(v, int):
            out += f"{v}"
        elif isinstance(v, np.bool_):
            out += f"{v.item()}"
        elif isinstance(v, torch.Tensor):
            out += f"{v.shape}"
        elif isinstance(v, dict):
            print_dict(v)
        print(out)
    print('}\n')


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


def draw_lidar_frame_(pc_range, resolution, ax_):
    lidar_frame = np.array([
        [0, 0],  # origin
        [3, 0],  # x-axis
        [0, 3]  # y-axis
    ])
    lidar_frame_in_bev = np.floor((lidar_frame - pc_range[:2]) / resolution)
    ax_.arrow(lidar_frame_in_bev[0, 0], lidar_frame_in_bev[0, 1],
              lidar_frame_in_bev[1, 0] - lidar_frame_in_bev[0, 0],
              lidar_frame_in_bev[1, 1] - lidar_frame_in_bev[0, 1],
              color='r', width=3)  # x-axis

    ax_.arrow(lidar_frame_in_bev[0, 0], lidar_frame_in_bev[0, 1],
              lidar_frame_in_bev[2, 0] - lidar_frame_in_bev[0, 0],
              lidar_frame_in_bev[2, 1] - lidar_frame_in_bev[0, 1],
              color='b', width=3)  # y-axis


def draw_boxes_in_bev_(boxes_in_bev, ax_: Axes, color='r'):
    """
    Args:
        boxes_in_bev (List[np.ndarray]): each box - (8, 2) forward: 0-1-2-3, backward: 4-5-6-7, up: 0-1-5-4
        ax_:
        color
    """
    for box_in_bev in boxes_in_bev:
        top_face = box_in_bev[[0, 1, 5, 4, 0]]
        ax_.plot(top_face[:, 0], top_face[:, 1], c=color)

        # draw heading
        center = (box_in_bev[0] + box_in_bev[5]) / 2.0
        mid_01 = (box_in_bev[0] + box_in_bev[1]) / 2.0
        heading_line = np.stack([center, mid_01], axis=0)
        ax_.plot(heading_line[:, 0], heading_line[:, 1], c=color)


def draw_lane_in_bev_(lane, pc_range, resolution, ax_, discretization_meters=1):
    """
    Args:
        lane (np.ndarray): (N, 3) - x, y, yaw in frame where BEV is generated (default: LiDAR frame)
    """
    lane_xy_in_bev = np.floor((lane[:, :2] - pc_range[:2]) / resolution)  # (N, 2)
    for _i in range(lane.shape[0]):
        cos, sin = discretization_meters * np.cos(lane[_i, -1]), discretization_meters * np.sin(lane[_i, -1])

        normalized_rgb_color = colorsys.hsv_to_rgb(np.rad2deg(lane[_i, -1]) / 360, 1., 1.)

        ax_.arrow(lane_xy_in_bev[_i, 0], lane_xy_in_bev[_i, 1], cos, sin, color=normalized_rgb_color, width=0.75)


def print_nuscenes_record(rec: dict, rec_type=None):
    print(f'--- {rec_type}' if rec_type is not None else '---')
    for k, v in rec.items():
        print(f"{k}: {v}")
    print('---\n')


def show_image_(ax_: Axes, img: np.ndarray, title=None, xlims=None, ylims=None, boxes=None, cmap=None):
    if cmap is None:
        ax_.imshow(img)
    else:
        ax_.imshow(img, cmap=cmap)
    ax_.set_aspect('equal')

    if title is not None:
        ax_.set_title(title)

    if xlims is not None:
        ax_.set_xlim(xlims)

    if ylims is not None:
        ax_.set_ylim(ylims)

    if boxes is not None:
        draw_boxes_in_bev_(boxes, ax_)
