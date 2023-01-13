from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from _dev_space.viz_tools import print_dict
from _dev_space._test.tools_4testing import load_data_to_tensor, load_dict_to_gpu, BackwardHook
from pcdet.models.detectors import SECONDNet, CenterPoint
import sys


def main(model_name, target_batch_idx=1, batch_size=1, is_training=True, use_correction=True):
    if model_name == 'second':
        cfg_file = './second_corrector_mini.yaml'
        ckpt = 'second_corrector_partA2_ep10.pth' if use_correction else 'second_corrector_partA2_nocorr_ep10.pth'
    elif model_name == 'pillar':
        raise ValueError("pointpillar hasn't been trained with 2nd stage")
        cfg_file = './pointpillars_corrector_mini.yaml'
    else:
        raise ValueError(f"{model_name} is unknown")

    cfg_from_yaml_file(cfg_file, cfg)

    cfg.MODEL.CORRECTOR.CORRECT_POINTS_WHILE_TRAINING = use_correction

    logger = common_utils.create_logger('./a_dummy_log.txt')
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                              batch_size=batch_size, dist=False, logger=logger, training=is_training,
                                              total_epochs=1, seed=666, workers=1)
    iter_dataloader = iter(dataloader)

    batch_dict = None
    for _ in range(target_batch_idx):
        batch_dict = next(iter_dataloader)
    load_data_to_tensor(batch_dict)
    print_dict(batch_dict, 'batch_dict')
    load_dict_to_gpu(batch_dict)

    if model_name == 'second':
        model = SECONDNet(cfg.MODEL, num_class=10, dataset=dataset)
    elif model_name == 'pillar':
        model = CenterPoint(cfg.MODEL, num_class=10, dataset=dataset)
    else:
        raise ValueError(f"{model_name} is unknown")
    print('---\n',
          model,
          '\n---')

    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    pred_dicts, recall_dicts = model(batch_dict)
    print_dict(pred_dicts[0], 'pred_dicts')
    print_dict(recall_dicts, 'recall_dicts')

    batch_dict.update({'pred_dicts': pred_dicts})
    print_dict(batch_dict, 'batch_dict')

    for k in (
    'encoded_spconv_tensor', 'encoded_spconv_tensor_stride', 'multi_scale_3d_features', 'multi_scale_3d_strides',
    'spatial_features',
    'spatial_features_stride', 'spatial_features_2d'):
        if k in batch_dict:
            batch_dict.pop(k)

    if batch_size == 1:
        sample_tk = batch_dict['metadata'][0]['token']
        print('sample_tk: ', sample_tk)
        torch.save(batch_dict, f"{model_name}_stage_pred_{sample_tk}.pth")
    else:
        torch.save(batch_dict, f"{model_name}_2stage_{int(use_correction)}corr_pred_batch_size{batch_size}.pth")


if __name__ == '__main__':
    # set this in the terminal:
    # CUDA_VISIBLE_DEVICES=2
    main(sys.argv[1], is_training=False, target_batch_idx=3, batch_size=10, use_correction=sys.argv[2] == 'use_correction')
