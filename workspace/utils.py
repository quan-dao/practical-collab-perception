import numpy as np
import torch


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



