import os
import importlib

import torch


def update_args(args):
    assert hasattr(args, 'cfg') and hasattr(args, 'local_rank')
    if args.cfg.endswith('.py'):
        args.cfg = args.cfg[:-3]
    args.cfg = args.cfg.replace('/', '.')
    cfg = importlib.import_module(args.cfg).cfg
    for k, v in cfg.items():
        setattr(args, k, v)

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)


def lr_autoscale(lr, base_total_bs, bs_per_gpu):
    return lr / base_total_bs * torch.cuda.device_count() * bs_per_gpu


def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,'
        'memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return int(total), int(used)


def occumpy_mem():
    total, used = check_mem(torch.cuda.current_device())
    max_mem = int(total * 0.95)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
