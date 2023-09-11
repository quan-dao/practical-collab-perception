import open3d as o3d
import numpy as np
import torch
from typing import Union, List
import matplotlib.pyplot as plt

from pcdet.datasets.nuscenes.nuscenes_temporal_utils import apply_se3_, make_se3


def get_boxes_vertices_coord(boxes: Union[np.ndarray, torch.Tensor]) -> List[np.ndarray]:
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
        ref_se3_box = make_se3(box[:3], yaw=box[6])
        apply_se3_(ref_se3_box, points_=vers)
        out.append(vers)

    return out


class PointsPainter(object):
    def __init__(self, 
                 xyz: Union[np.ndarray, torch.Tensor], 
                 boxes: Union[np.ndarray, torch.Tensor] = None):
        """
        Args:
            xyz: (N, 3)
            boxes: (N_b, 7[+2][+1]) - x, y, z, dx, dy, dz, yaw, [velo_x, velo_y], [class_idx]
        """
        self.xyz = xyz
        self.pointcloud = o3d.geometry.PointCloud()
        self.pointcloud.points = o3d.utility.Vector3dVector(xyz)
        
        if boxes is not None:
            boxes_vertices = get_boxes_vertices_coord(boxes)
            self.boxes_center = boxes[:, :3]
            self.boxes = [self.create_cube(vers) for vers in boxes_vertices]
            
            self.num_boxes = boxes.shape[0]
        else:
            self.boxes_center, self.boxes, self.num_boxes = None, None, None

        self.lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=np.zeros(3))
    
    @staticmethod
    def create_cube(vers: np.ndarray):
        # vers: (8, 3)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # front
            [4, 5], [5, 6], [6, 7], [7, 4],  # back
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting front & back
            [0, 2], [1, 3]  # denote forward face
        ]
        cube = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vers),
            lines=o3d.utility.Vector2iVector(lines),
        )
        
        return cube
    
    def _color_point_cloud(self, points_color: np.ndarray):
        self.pointcloud.colors = o3d.utility.Vector3dVector(points_color)
        
    def _color_boxes(self, boxes_color: np.ndarray):
        num_edges = 14
        for b_idx in range(len(self.boxes)):
            color = [boxes_color[b_idx] for _ in range(num_edges)]
            self.boxes[b_idx].colors = o3d.utility.Vector3dVector(color)

    def _draw_boxes_velocity(self, boxes_velo: np.ndarray):
        assert boxes_velo.shape[1] == 2, f"boxes_velo.shape[1]: {boxes_velo.shape[1]}"
        all_displacement_lines = []
        for b_idx in range(self.boxes_center.shape[0]):
            tip = [self.boxes_center[b_idx, _i] + boxes_velo[b_idx, _i] * 0.5 for _i in range(2)]  # [x, y]
            tip.append(self.boxes_center[b_idx, 2])  # z
            pts = np.stack([self.boxes_center[b_idx], np.array(tip)], axis=0)
            o3d_displacement_line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            all_displacement_lines.append(o3d_displacement_line)
        return all_displacement_lines
    
    def _draw_points_offset(self, points_offset: np.ndarray):
        mask_has_offset = np.square(points_offset).sum(axis=1) > 0.2**2
        xyz_ = self.xyz[mask_has_offset]
        offset_ = points_offset[mask_has_offset]
        
        all_o3d_offsets = []
        
        for i in range(xyz_.shape[0]):
            pts = np.stack([
                xyz_[i],
                xyz_[i] + offset_[i]
            ], axis=0)
            o3d_offset = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            o3d_offset.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]]))
            all_o3d_offsets.append(o3d_offset)

        return all_o3d_offsets
    
    def _draw_special_points(self, xyz: np.ndarray):
        assert xyz.shape[1] == 3, f"{xyz.shape[1]} != 3"
        out = list()
        for i in range(xyz.shape[0]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
            sphere.translate(xyz[i].reshape(3, 1))
            sphere.paint_uniform_color(np.array([255. / 255., 102. / 255., 0]))
            out.append(sphere)
        
        return out

    def show(self,
             xyz_color: np.ndarray = None, 
             boxes_color: np.ndarray = None, **kwargs) -> None:
        
        if xyz_color is not None:
            self._color_point_cloud(xyz_color)

        objects_to_draw = [self.pointcloud, self.lidar_frame]

        if self.boxes is not None :
            if boxes_color is None:
                # set detail color for boxes
                boxes_color = np.tile(np.array([1, 0, 0]).reshape(1, 3), (self.num_boxes, 1))

            self._color_boxes(boxes_color)

            objects_to_draw += self.boxes

        if 'boxes_velo' in kwargs:
            o3d_displacement_lines = self._draw_boxes_velocity(kwargs['boxes_velo'])  # (N_box, 2)
            objects_to_draw += o3d_displacement_lines

        if 'points_offset' in kwargs:
            # kwargs['points_offset']: (N_pts, 3)
            o3d_points_offset = self._draw_points_offset(kwargs['points_offset'])
            objects_to_draw += o3d_points_offset
        
        if 'special_points' in kwargs:
            o3d_special_points = self._draw_special_points(kwargs['special_points'])
            objects_to_draw += o3d_special_points

        o3d.visualization.draw_geometries(objects_to_draw)


def print_dict(d: dict, name=''):
    print(f'{name}', ': {')
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
    print('} eod ', name)


def color_points_binary(points_binary_channel: np.ndarray):
    assert points_binary_channel.shape == (points_binary_channel.shape[0],), f"{points_binary_channel.shape} != {(points_binary_channel.shape[0],)}"
    points_color = np.zeros((points_binary_channel.shape[0], 3))
    points_color[points_binary_channel.astype(int) == 1, 0] = 1.0  # red for 1, black for 0
    return points_color


class BEVPainter:
    def __init__(self, pc_range: np.ndarray) -> None:
        self.pc_range = pc_range

    def draw_1box(self, axe, box: np.ndarray, box_color, linestyle='solid', linewidth=1):
        vertices = np.array([
            [1,   1],
            [-1,  1],
            [-1, -1],
            [1, -1],
        ]) / 2.0
        vertices *= box[3: 5]  # (4, 2)
        vertices = np.pad(vertices, pad_width=[(0, 0), (0, 1)], constant_values=1.)  # (4, 3)

        cos, sin = np.cos(box[-1]), np.sin(box[-1])
        box_to_lidar = np.array([
            [cos,  -sin, box[0]],
            [sin,  cos,  box[1]],
            [0.,   0.,   1.]
        ])
        vertices = vertices @ box_to_lidar.T  # (4, 3)

        axe.plot(vertices[[0, 1, 2, 3, 0], 0], vertices[[0, 1, 2, 3, 0], 1], c=box_color, linestyle=linestyle, linewidth=linewidth)

    def show_bev(self, xyz: np.ndarray, axe=None, xyz_color=None, special_points: np.ndarray = None, 
                 gt_boxes=None, gt_boxes_color=None, show_gt_idx=False, invisible_gt_indices=list(),
                 pred_boxes=None, pred_boxes_color=None):
        if axe is None:
            fig, axe = plt.subplots()
        
        if xyz_color is None:
            xyz_color = np.zeros((xyz.shape[0], 3))
            xyz_color[:, 2] = 1.  # blue

        axe.scatter(xyz[:, 0], xyz[:, 1], c=xyz_color, s=5.)

        if special_points is not None:
            axe.scatter(special_points[:, 0], special_points[:, 1], c='darkorange', marker='*', s=25.0)

        if gt_boxes is not None:
            assert gt_boxes.shape[1] == 7, f"{gt_boxes.shape[1]} != 7"
            if gt_boxes_color is None:
                gt_boxes_color = np.zeros((gt_boxes.shape[0], 3))
                gt_boxes_color[:, 0] = 1.

            for bidx, box in enumerate(gt_boxes):
                self.draw_1box(axe, box, gt_boxes_color[bidx], 
                               linestyle='solid' if bidx not in invisible_gt_indices else 'dashed',
                               linewidth=2)
                if show_gt_idx:
                    axe.text(box[0], box[1], f"gt_{bidx}")

        if pred_boxes is not None:
            assert pred_boxes.shape[1] == 7, f"{pred_boxes.shape[1]} != 7"
            if pred_boxes_color is None:
                pred_boxes_color = np.zeros((pred_boxes.shape[0], 3))
                pred_boxes_color[:, 0] = 1.

            for bidx, box in enumerate(pred_boxes):
                self.draw_1box(axe, box, pred_boxes_color[bidx])
        
        axe.set_xlim([self.pc_range[0], self.pc_range[3]])
        axe.set_ylim([self.pc_range[1], self.pc_range[4]])
        axe.set_aspect('equal', 'box')
        axe.set_xticks([])
        axe.set_yticks([])
