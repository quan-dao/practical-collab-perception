import torch
from pcdet.utils.common_utils import rotate_points_along_z
from einops import rearrange


def get_axis_aligned_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """
    Args:
        boxes: (N, 7 + C) - x, y, z, dx, dy, dz, angle, [velo_x, velo_y, class]
    Returns:
        axis_aligned_boxes: (N, 4) - x_min, y_min, x_max, y_max (in world frame)
    """
    corners = torch.tensor([
        [1, 1, -1, -1],  # x in normalized box coord
        [-1, 1, 1, -1],  # y in normalized box coord
        [1, 1, 1, 1]  # z in normalized box coord
    ]).view(4, 3).float().to(boxes.device)
    # map corners from boxes' local frame to world frame
    corners = 0.5 * rearrange(boxes[:, 3: 6], 'N C -> N 1 C') * rearrange(corners, 'M C -> 1 M C')  # (N, 4, 3) - in box coord
    corners = rotate_points_along_z(corners, boxes[:, 6]) + rearrange(boxes[:, :3], 'N C -> N 1 C')  # (N, 4, 3) - in world coord
    # get axis-aligned boxes
    min_xy = torch.min(corners[:, :, :-1], dim=1)[0]  # (N, 2)
    max_xy = torch.max(corners[:, :, :-1], dim=1)[0]  # (N, 2)
    aa_boxes = torch.cat([min_xy, max_xy], dim=1)  # (N, 4) - x_min, y_min, x_max, y_max
    return aa_boxes
    

def get_axis_aligned_box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Args:
        boxes: (N, 4) - x_min, y_min, x_max, y_max
    Returns:
        area: (N,)
    """
    area = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.)
    return area


def get_axis_aligned_iou(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
    """
    One-to-one axis-aligned IoU

    Args:
        boxes_1: (N, 7 + C) - x, y, z, dx, dy, dz, angle, [velo_x, velo_y, class]
        boxes_2: (N, 7 + C) - x, y, z, dx, dy, dz, angle, [velo_x, velo_y, class]
    Returns:
        iou: (N,)
    """
    assert boxes_1.shape[0] == boxes_2.shape[0], f"expect boxes_1.shape[0] == boxes_2.shape[0]; get {boxes_1.shape[0]} != {boxes_2.shape[0]}"
    aa_box_1 = get_axis_aligned_boxes(boxes_1)  # (N, 4) - x_min, y_min, x_max, y_max
    aa_box_2 = get_axis_aligned_boxes(boxes_2)  # (N, 4) - x_min, y_min, x_max, y_max

    inter_xmax = torch.minimum(aa_box_1[:, 2], aa_box_2[:, 2])
    inter_ymax = torch.minimum(aa_box_1[:, 3], aa_box_2[:, 3])
    inter_xmin = torch.maximum(aa_box_1[:, 0], aa_box_2[:, 0])
    inter_ymin = torch.maximum(aa_box_1[:, 1], aa_box_2[:, 1])

    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0.) * torch.clamp(inter_ymax - inter_ymin, min=0.)

    area_1 = get_axis_aligned_box_area(aa_box_1)
    area_2 = get_axis_aligned_box_area(aa_box_2)
    union = area_1 + area_2 - inter_area
    
    nonzero_union = union > 0
    iou = torch.zeros_like(inter_area)
    iou[nonzero_union] = inter_area[nonzero_union] / union[nonzero_union]
    return iou
