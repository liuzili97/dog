import os
import importlib
from tabulate import tabulate
from termcolor import colored

import torch

__all__ = ['describe_model', 'update_args', 'lr_autoscale', 'occumpy_mem', 'get_free_gpu']


def describe_model(model):
    headers = ['name', 'shape', '#elements', '(M)', '(MB)', 'trainable', 'dtype']

    data = []
    trainable_count = 0
    total = 0
    total_size = 0
    trainable_total = 0

    for name, param in model.named_parameters():
        dtype = str(param.data.dtype)
        param_mb = 'NAN'
        param_bype = float(''.join([s for s in dtype if s.isdigit()])) / 8

        total += param.data.nelement()
        if param_bype:
            param_mb = '{:.02f}'.format(param.data.nelement() / 1024.0 ** 2 * param_bype)
            total_size += param.data.nelement() * param_bype
        data.append([name.replace('.', '/'), list(param.size()),
                     '{0:,}'.format(param.data.nelement()),
                     '{:.02f}'.format(param.data.nelement() / 1024.0 ** 2), param_mb,
                     param.requires_grad, dtype])
        if param.requires_grad:
            trainable_count += 1
            trainable_total += param.data.nelement()

    table = tabulate(data, headers=headers)

    summary_msg = colored(
        "\nNumber of All variables: {}".format(len(data)) +
        "\nAll parameters (elements): {:.02f}M".format(total / 1024.0 ** 2) +
        "\nAll parameters (size): {:.02f}MB".format(total_size / 1024.0 ** 2) +
        "\nNumber of Trainable variables: {}".format(trainable_count) +
        "\nAll trainable parameters (elements): {:.02f}M\n".format(
            trainable_total / 1024.0 ** 2), 'cyan')
    print(colored("List of All Variables: \n", 'cyan') + table + summary_msg)


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
    return lr / base_total_bs * torch.distributed.get_world_size() * bs_per_gpu


def occumpy_mem():
    total, used = _check_mem(torch.cuda.current_device())
    max_mem = int(total * 0.95)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


def get_free_gpu(thre=0.9):
    qargs = ['index', 'memory.free', 'memory.total', 'power.draw']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    gpus_info = [_parse(line, qargs) for line in results]
    gpu_num = len(gpus_info)
    gpu_free = [False for _ in range(gpu_num)]
    for gpu_info in gpus_info:
        gpu_id = int(gpu_info['index'])
        gpu_free[gpu_id] = float(gpu_info['memory.free']) / float(gpu_info['memory.total']) > thre
    gpu_free_id = [str(idx) for idx in range(gpu_num) if gpu_free[idx]]
    return gpu_free_id, gpu_num


def _check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,'
        'memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return int(total), int(used)


def _parse(line, qargs):
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
    power_manage_enable = lambda v: (not 'Not Support' in v)
    to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').replace('W', ''))
    process = lambda k, v: ((int(to_numberic(v)) if power_manage_enable(v) else 1)
                            if k in numberic_args else v.strip())
    return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}
