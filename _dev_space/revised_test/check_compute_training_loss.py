from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from _dev_space.viz_tools import print_dict
from _dev_space._test.tools_4testing import load_data_to_tensor, load_dict_to_gpu, BackwardHook
from pcdet.models.detectors import SECONDNet, CenterPoint
import sys


def main(model_name, target_batch_idx=1, batch_size=1, is_training=True):
    if model_name == 'second':
        cfg_file = './second_corrector_mini.yaml'
    elif model_name == 'pillar':
        cfg_file = './pointpillars_corrector_mini.yaml'
    else:
        raise ValueError(f"{model_name} is unknown")

    cfg_from_yaml_file(cfg_file, cfg)
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

    model.cuda()
    bw_hooks = [BackwardHook(name, param) for name, param in model.named_parameters()]

    ret_dict, tb_dict, disp_dict = model(batch_dict)
    print_dict(tb_dict, 'tb_dict')

    model.zero_grad()
    ret_dict['loss'].backward()
    for hook in bw_hooks:
        if hook.grad_mag < 1e-4:
            print(f'zero grad at {hook.name}')


if __name__ == '__main__':
    # set this in the terminal:
    # CUDA_VISIBLE_DEVICES=2
    main(sys.argv[1])
