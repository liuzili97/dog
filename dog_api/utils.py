import os
import re
import importlib
import socket
from tabulate import tabulate
from termcolor import colored

import torch

__all__ = ['describe_model', 'update_args', 'lr_autoscale',
           'occupy_mem', 'get_free_gpu', 'get_slurm_scontrol_show']


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


def occupy_mem():
    this_device = torch.cuda.current_device()
    print(colored(f"Occupying {socket.gethostname()}:{this_device}", 'red'))
    _, free = check_mem(this_device)
    block_mem = int(free * 0.95)
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


def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,'
        'memory.free --format=csv,nounits,noheader').read().strip().split("\n")
    total, free = devices_info[int(cuda_device)].split(',')
    return int(total), int(free)


def _parse(line, qargs):
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
    power_manage_enable = lambda v: (not 'Not Support' in v)
    to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').replace('W', ''))
    process = lambda k, v: ((int(to_numberic(v)) if power_manage_enable(v) else 1)
                            if k in numberic_args else v.strip())
    return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}


def get_slurm_scontrol_show(node_name):
    slurm_info = dict(
        all_cpu=0, all_mem=0, all_gpu=0,
        alloc_cpu=0, alloc_mem=0, alloc_gpu=0,
        partitions=''
    )
    data = os.popen(f'scontrol show node {node_name}').read()
    all_tres = re.compile('CfgTRES=(.*)').search(data)
    alloc_tres = re.compile('AllocTRES=(.*)').search(data)
    partitions = re.compile('Partitions=(.*)').search(data)
    if all_tres:
        all_tres = all_tres[1].replace(',', '\n')
        all_cpu = re.compile('cpu=(.*)').search(all_tres)
        all_mem = re.compile('mem=(.*)M').search(all_tres)
        all_gpu = re.compile('gres/gpu=(.*)').search(all_tres)
        if all_cpu:
            slurm_info['all_cpu'] = int(all_cpu[1])
        if all_mem:
            slurm_info['all_mem'] = int(all_mem[1])
        if all_gpu:
            slurm_info['all_gpu'] = int(all_gpu[1])

    if alloc_tres:
        alloc_tres = alloc_tres[1].replace(',', '\n')
        alloc_cpu = re.compile('cpu=(.*)').search(alloc_tres)
        alloc_mem = re.compile('mem=(.*)M').search(alloc_tres)
        alloc_gpu = re.compile('gres/gpu=(.*)').search(alloc_tres)
        if alloc_cpu:
            slurm_info['alloc_cpu'] = int(alloc_cpu[1])
        if alloc_mem:
            slurm_info['alloc_mem'] = int(alloc_mem[1])
        if alloc_gpu:
            slurm_info['alloc_gpu'] = int(alloc_gpu[1])

    if partitions:
        slurm_info['partitions'] = partitions[1]
    return slurm_info


def get_slurm_squeue(all_nodes):
    nodes_users = {}
    for node_name in all_nodes:
        nodes_users[node_name] = set()
    data = os.popen(f'squeue').readlines()
    user_nodes = []
    for line in data:
        line = line.strip().split()
        if line[-1].startswith('NODELIST') or line[-1].startswith('('):
            continue
        user_nodes.append([line[-5], line[-1]])
    for user, node_name in user_nodes:
        if node_name in nodes_users:
            nodes_users[node_name].add(user)
        else:
            node_ids = re.compile("(.*)\[(.*)\]").search(node_name)
            if node_ids:
                node_prefix, node_ids = node_ids[1], node_ids[2]
                node_ids = get_node_ids(node_ids)
                node_names = [node_prefix + i for i in node_ids]
                for s_node_name in node_names:
                    nodes_users[s_node_name].add(user)
    for k, v in nodes_users.items():
        users = sorted(list(v))
        for i, u in enumerate(users):
            if os.environ['USER'].startswith(u):
                users[i] = colored(u, attrs=['bold'])
        nodes_users[k] = users
    return nodes_users


def get_node_ids(node_str):
    res = []
    node_ids = [node_id for node_id in node_str.split(',')]
    for i, node_id in enumerate(node_ids):
        if '-' in node_id:
            start, end = node_id.split('-')
            res.extend([str(i) for i in range(int(start), int(end) + 1)])
        else:
            res.append(node_id)
    return res
