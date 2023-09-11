import numpy as np
import torch
import matplotlib.pyplot as plt
from workspace.o3d_visualization import PointsPainter, print_dict, BEVPainter
from pcdet.datasets.nuscenes.nuscenes_temporal_utils import apply_se3_


def main(filename_batch_dict: str, invisible_gt_indices: list, saveimg=False):
    batch_dict = torch.load(filename_batch_dict, map_location=torch.device('cpu'))
    print_dict(batch_dict, name='batch_dict')

    points = batch_dict['points']
    gt_boxes = batch_dict['gt_boxes'][0][:, :7].numpy()
    
    pred_dict = batch_dict['final_box_dicts'][0]
    pred_boxes = pred_dict['pred_boxes'].numpy()
    pred_scores = pred_dict['pred_scores'].numpy()
    
    # separate points & modar
    mask_modar = points[:, -3] > 0
    modar = points[mask_modar]  # (M, 14)
    print('modar: ', modar.shape)
    ego_points = points[torch.logical_not(mask_modar)]  # (N, 14)

    # agent points
    agent_points = list()
    for agent_id, agent_dict in batch_dict['exchange'].items():
        ag_pts = agent_dict['points']
        apply_se3_(agent_dict['ego_se3_agent'], points_=ag_pts)
        agent_points.append(ag_pts)

    agent_points = np.concatenate(agent_points)  # (N_a, 4) - x, y, z, reflectant

    # ========================================================================
    # painter = PointsPainter(ego_points[:, 1: 4])
    
    # color_ego_points = torch.zeros(ego_points.shape[0], 3)
    # color_ego_points[:, 2] = 1.
    # painter.show(xyz_color=color_ego_points, special_points=modar[:, 1: 4])

    # ========================================================================
    fig, axe = plt.subplots(figsize=(20, 20))
    f1, ax1 = plt.subplots(figsize=(20, 20))

    mask_present = ego_points[:, -2] == torch.max(ego_points[:, -2])
    ego_points = ego_points[mask_present, 1: 4]
    
    mask_valid_pred = pred_scores > 0.3
    pred_boxes = pred_boxes[mask_valid_pred]

    colors_gt_boxes = np.zeros((gt_boxes.shape[0], 3))
    colors_gt_boxes[:, 1] = 1.  # green -> gt
    
    colors_pred_boxes = np.zeros((pred_boxes.shape[0], 3))
    colors_pred_boxes[:, 0] = 1.  # pred -> gt

    points_all = np.concatenate([ego_points, agent_points[:, :3]])
    colors_points_all = np.zeros((points_all.shape[0], 3))
    colors_points_all[:torch.sum(mask_present), 2] = 1.0  # blue for ego
    colors_points_all[-agent_points.shape[0]:] = np.array([0.5, 0.5, 0.5])  # gray for agents

    bev_painter = BEVPainter([-51.2, -51.2, -8.0, 51.2, 51.2, 0.0])
    bev_painter.show_bev(points_all, axe, xyz_color=colors_points_all, special_points=modar[:, 1: 4], 
                         gt_boxes=gt_boxes, gt_boxes_color=colors_gt_boxes, invisible_gt_indices=invisible_gt_indices,
                         pred_boxes=pred_boxes, pred_boxes_color=colors_pred_boxes)
    
    bev_painter.show_bev(ego_points, ax1, xyz_color=colors_points_all[:torch.sum(mask_present)], 
                         gt_boxes=gt_boxes, gt_boxes_color=colors_gt_boxes, show_gt_idx=True)
    
    if saveimg:
        figname = filename.split('.')[0]
        fig.savefig(f'{figname}.png')

    plt.show()


if __name__ == '__main__':
    invisible_dict = {
        'for_quali_gxyl78y304i5637k6mq376abyxgeharx.pth': [8, 9],
        'for_quali_p9572g7z415y2p0ka6na6tzpyziz5sof.pth': [],
        'for_quali_2269615u3uku5rbrgn37ux4iqi44q0hf.pth': [5, 12, 13],  # - 5 is occluded
        'for_quali_3wq3h16h60upojri1q684ya1c371152r.pth': [],
        'for_quali_p2bt96217b00195wp301rl3m8f035877.pth': [4,],
        'for_quali_j31xj68tr99kont9b3ui8607s71w25fw.pth': [10, 1, 11, 16, 14, 4, 3, 5],
    }
    # filename = 'for_quali_gxyl78y304i5637k6mq376abyxgeharx.pth'  # invisible: [8, 9]
    # filename = 'for_quali_p9572g7z415y2p0ka6na6tzpyziz5sof.pth'  # invisible: []
    # filename = 'for_quali_2269615u3uku5rbrgn37ux4iqi44q0hf.pth'  # invisible: [5, 12, 13] - 5 is occluded
    # filename = 'for_quali_3wq3h16h60upojri1q684ya1c371152r.pth'  # invisible: [] - a lot of FP in MoDAR
    # filename = 'for_quali_8td8g5oce50a322cw9kddo303ljn1129.pth'  # invisible: [1, 9, 8, 13, 0, 2, 6, 16]
    # filename = 'for_quali_p2bt96217b00195wp301rl3m8f035877.pth'  # invisible: [4,]
    # filename = 'for_quali_j31xj68tr99kont9b3ui8607s71w25fw.pth'
    filename = 'for_quali_p2bt96217b00195wp301rl3m8f035877' + '.pth'
    invisible_gt_indices = invisible_dict[filename]
    main(filename, invisible_gt_indices=invisible_gt_indices, saveimg=True)
