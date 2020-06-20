import os
import yaml
import subprocess
import functools

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def set_dog_base():
    if 'DOG_BASE' not in os.environ:
        os.environ['DOG_BASE'] = os.path.join(os.path.dirname(__file__), '..')


def init_dog_dir():
    with open(os.path.join(os.environ['DOG_BASE'], 'cfg', 'setting.yml')) as f:
        cfg = yaml.safe_load(f)
    dog_dirname = cfg['DOG_DIRNAME']

    # init dog file
    dog_dir = os.path.join(os.environ['HOME'], dog_dirname)
    if not os.path.isdir(dog_dir):
        os.makedirs(dog_dir)

    os.environ['DOG_DIRNAME'] = dog_dirname
    os.environ['DOG_DIR'] = dog_dir


def set_dog_shell(use_slurm):
    if use_slurm:
        os.environ['DOG_SHELL'] = 'slurm'
    else:
        os.environ['DOG_SHELL'] = 'dist'


def set_dog_key(key):
    # set in register
    os.environ['DOG_KEY'] = key
    os.environ['DOG_FILE'] = os.path.join(os.environ['DOG_DIR'], key)
    os.environ['DOG_DEVICEDIR'] = os.path.join(os.environ['DOG_DIR'], f".{key}")


def set_target_env(entry_path, entry_key_dir):
    # set in main
    os.environ['ENTRY_PATH'] = entry_path
    os.environ['ENTRY_KEY_DIR'] = entry_key_dir


def is_dog_launched():
    return 'DOG_KEY' in os.environ


def dog_launched(old_func):
    @functools.wraps(old_func)
    def new_func(*args, **kwargs):
        if not is_dog_launched():
            print("NOT LAUNCHED")
            return

        output = old_func(*args, **kwargs)
        return output
    return new_func


def set_env(env_dict):
    for k, v in env_dict.items():
        os.environ[k] = str(v)


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def init_dist_pytorch(backend='nccl', **kwargs):
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def init_dist_slurm(backend='nccl', port=29500, **kwargs):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    addr = subprocess.getoutput(
        'scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    dist.init_process_group(backend=backend)
