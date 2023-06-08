import torch
from pathlib import Path
from time import time
import subprocess
import lovely_tensors as lt

from tools import build_test_infra, to_tensor
from workspace.o3d_visualization import PointsPainter, print_dict


def main(target_dataloader_idx=3, training=False, batch_size=2):
    cfg_file = './cfgs/uda_pp_basic.yaml'
    ckpt_dir = Path('/home/jupyter-dao-mq/workspace/learn-to-align-bev/output/cfgs/nuscenes_models/uda_pp_basic/def/ckpt')
    model, dataloader = build_test_infra(cfg_file, 
                                         training=training, 
                                         batch_size=batch_size, 
                                         ckpt_file=ckpt_dir / 'checkpoint_epoch_20.pth')

    # print('---------------------\n', 
    #       model, 
    #       '\n---------------------')

    iter_dataloader = iter(dataloader)
    data_time, counter = 0, 0
    for _ in range(target_dataloader_idx):
        start = time.time()
        batch_dict = next(iter_dataloader)
        data_time += (time.time() - start)
        counter += 1

    print_dict(batch_dict, 'batch_dict')
    print('avg data time: ', data_time / float(counter))

    to_tensor(batch_dict, move_to_gpu=True)
    model.cuda()

    model.eval()
    with torch.no_grad():
        pred_dicts, recall_dicts = model(batch_dict)

    batch_dict['pred_dicts'] = pred_dicts
    torch.save(batch_dict, 'artifact/uda_pp_batch_dict.pth')


def show_batch_dict(copy_from_azog: bool, chosen_batch_idx: int):
    if copy_from_azog:
        domain = 'jupyter-dao-mq@azog.ls2n.ec-nantes.fr'
        root_at_domain = '/home/jupyter-dao-mq/workspace/learn-to-align-bev/test_space/artifact'
        src_file = f'{domain}:{root_at_domain}/uda_pp_batch_dict.pth'
        cmd_out = subprocess.run(['scp', src_file, './artifact/'], stdout=subprocess.PIPE)

    batch_dict = torch.load('./artifact/uda_pp_batch_dict.pth', map_location=torch.device('cpu'))
    print_dict(batch_dict)

    points = batch_dict['points']
    pred_dicts = batch_dict['pred_dicts']
    gt_boxes = batch_dict['gt_boxes'][chosen_batch_idx]
    
    mask_1sample = points[:, 0].long() == chosen_batch_idx
    points = points[mask_1sample]
    
    pred_dict = pred_dicts[chosen_batch_idx]
    print_dict(pred_dict, 'pred_dict')
    pred_boxes = pred_dict['pred_boxes']
    pred_scores = pred_dict['pred_scores']
    print('pred_scores:\n', pred_scores)

    boxes = torch.cat([gt_boxes[:, :7], pred_boxes[:, :7]], dim=0)
    boxes_color = boxes.new_zeros(boxes.shape[0], 3)
    boxes_color[:gt_boxes.shape[0], 0] = 1  # gt = red
    boxes_color[gt_boxes.shape[0]:, 2] = 1  # pred = blue

    painter = PointsPainter(xyz=points[:, 1: 4], boxes=boxes)
    painter.show(boxes_color=boxes_color)
    # painter = PointsPainter(xyz=points[:, 1: 4], boxes=pred_boxes)
    # pred_boxes_color = pred_boxes.new_zeros(pred_boxes.shape[0], 3)
    # pred_boxes_color[:, 2] = 1.0
    # painter.show(boxes_color=pred_boxes_color)


if __name__ == '__main__':
    lt.monkey_patch()
    # main(target_dataloader_idx=3,
    #      training=True,
    #      batch_size=10)

    show_batch_dict(copy_from_azog=False,
                    chosen_batch_idx=1)
