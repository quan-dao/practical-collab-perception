import torch
import time
from pathlib import Path
import lovely_tensors as lt
import argparse
from copy import deepcopy

from tools import build_test_infra, to_tensor
from workspace.o3d_visualization import PointsPainter, print_dict


def inference(training: bool, 
              batch_size: int, 
              target_dataloader_idx: int,
              ckpt_path: str = '',
              print_model: bool = False):
    cfg_file = './cfgs/uda_pointpillar_basic.yaml'
    model, dataloader = build_test_infra(cfg_file, 
                                         training=training, 
                                         batch_size=batch_size, 
                                         ckpt_file=ckpt_path if ckpt_path != '' else None)

    if print_model:
        print('---------------------\n', 
            model, 
            '\n---------------------')

    iter_dataloader = iter(dataloader)
    data_time, counter = 0, 0
    for _ in range(target_dataloader_idx):
        start = time.time()
        batch_dict = next(iter_dataloader)
        data_time += (time.time() - start)
        counter += 1

    input_dict = deepcopy(batch_dict)
    to_tensor(input_dict)

    print_dict(batch_dict, 'batch_dict')
    print('avg data time: ', data_time / float(counter))

    # move to gpu
    to_tensor(batch_dict, move_to_gpu=True)
    model.cuda()

    # inference
    model.eval()
    with torch.no_grad():
        pred_dicts, recall_dicts = model(batch_dict)

    output_dict = {'pred_dicts': pred_dicts, 'input_dict': input_dict}
    torch.save(output_dict, 'artifact/uda_pointpillar_basic_output.pth')
    print('output is saved to artifact/uda_pointpillar_basic_output.pth')


def viz_result(output_dict_path: str,
               chosen_batch_dict: int = -1):
    output_dict = torch.load(output_dict_path, map_location=torch.device('cpu'))
    input_dict = output_dict['input_dict']
    pred_dicts = output_dict['pred_dicts']

    points = input_dict['points']
    
    gt_boxes = input_dict['gt_boxes']
    
    classes_color = torch.tensor([
        [1, 0, 0],  # red - car
        [0, 0, 1]  # blue - ped
    ]).float()

    _batch_indices = range(input_dict['batch_size']) if chosen_batch_dict == -1 else range(chosen_batch_dict, chosen_batch_dict + 1)
    for btc_idx in _batch_indices:
        mask_pts_now = points[:, 0].long() == btc_idx
        pts = points[mask_pts_now]

        gt_boxes_now = gt_boxes[btc_idx]

        pred_dict_now = pred_dicts[btc_idx]
        print_dict(pred_dict_now, 'pred_dict_now')
        
        pred_boxes = pred_dict_now['pred_boxes']
        pred_labels = pred_dict_now['pred_labels']
        print(f"pred_labels | min: {pred_labels.min()} | max: {pred_labels.max()}")

        disco_box_now = torch.from_numpy(input_dict['metadata'][btc_idx]['disco_boxes']).float()
        print(f'disco_box_now ({type(disco_box_now)}): ', disco_box_now.shape)

        boxes = torch.cat([gt_boxes_now[:, :7], pred_boxes, disco_box_now[:, :7]])

        # ---
        painter = PointsPainter(pts[:, 1: 4], boxes)
        
        gt_boxes_color = torch.tensor([0, 1, 0]).float().unsqueeze(0).repeat(gt_boxes_now.shape[0], 1)
        pred_boxes_color = classes_color[pred_labels - 1]
        disco_box_color = torch.tensor([1.0, 0.64, 0.]).float().unsqueeze(0).repeat(disco_box_now.shape[0], 1)

        boxes_color = torch.cat([gt_boxes_color, pred_boxes_color, disco_box_color])
        painter.show(boxes_color=boxes_color)


def main(use_inference: bool,
         use_viz_result: bool, 
         training: bool, 
         batch_size: int, 
         target_dataloader_idx: int,
         ckpt_path: str = '',
         print_model: bool = False,
         output_dict_path: str = '',
         chosen_batch_dict: int = -1):
    if use_inference:
        inference(training, batch_size, target_dataloader_idx, ckpt_path, print_model)
    
    if use_viz_result:
        viz_result(output_dict_path, chosen_batch_dict)


if __name__ == '__main__':
    lt.monkey_patch()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--use_inference', type=int)
    parser.add_argument('--use_viz_result', type=int)
    parser.add_argument('--training', type=int)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--target_dataloader_idx', type=int, default=3)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--print_model', type=int, default=0)
    parser.add_argument('--output_dict_path', type=str, default='')
    parser.add_argument('--chosen_batch_dict', type=int, default=-1)

    args = parser.parse_args()
    main(args.use_inference == 1,
         args.use_viz_result == 1, 
         args.training == 1, 
         args.batch_size, 
         args.target_dataloader_idx,
         args.ckpt_path,
         args.print_model == 1,
         args.output_dict_path,
         args.chosen_batch_dict)
