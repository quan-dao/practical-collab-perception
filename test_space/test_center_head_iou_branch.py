from tools import build_test_infra, to_tensor
from workspace.utils import print_dict


def main():
    cfg_file = './cfgs/pointpillar_jr.yaml'
    model, dataloader = build_test_infra(cfg_file, ckpt_file='./ckpts/cbgs_pp_centerpoint_nds6070.pth', 
                                        training=False, batch_size=2)
    
    iter_dataloader = iter(dataloader)
    for _ in range(3):
        batch_dict = next(iter_dataloader)
    
    to_tensor(batch_dict, move_to_gpu=True)
    
    model.cuda()
    model.train()
    ret_dict, tb_dict, disp_dict = model(batch_dict)
    print('loss: ', ret_dict['loss'])


if __name__ == '__main__':
    main()
