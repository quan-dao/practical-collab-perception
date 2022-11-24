import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.models.detectors import Aligner

from tools_4testing import load_data_to_tensor, load_dict_to_gpu, BackwardHook
from _dev_space.viz_tools import print_dict


np.random.seed(666)


def main(**kwargs):
    cfg_file = './tail_cutter_cfg.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger('./dummy_log.txt')
    dataset, dataloader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=2,
                                              dist=False, logger=logger, training=False, total_epochs=1, seed=666)
    iter_dataloader = iter(dataloader)
    batch_dict = None
    for _ in range(kwargs.get('chosen_iter', 5)):
        batch_dict = next(iter_dataloader)
    load_data_to_tensor(batch_dict)

    model = Aligner(cfg.MODEL, num_class=10, dataset=dataset)
    print('---')
    print(model)
    print('---')

    model.cuda()
    load_dict_to_gpu(batch_dict)

    bw_hooks = [BackwardHook(name, param, is_cuda=True) for name, param in model.named_parameters()]
    ret_dict, tb_dict, _ = model(batch_dict)
    loss = ret_dict['loss']
    model.zero_grad()
    loss.backward()

    print('---')
    print_dict(tb_dict)
    print('---')

    for hook in bw_hooks:
        if hook.grad_mag < 1e-4:
            print(f'zero grad at {hook.name}')


if __name__ == '__main__':
    main()
