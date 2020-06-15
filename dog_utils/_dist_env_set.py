import os
import socket
import yaml

hostname = socket.gethostname()


def set_slurm_env(hostname):
    gpu_info = dict()
    for i in range(1, 7):
        gpu_info['node{}'.format(i)] = dict(name='w2080ti', gpus_per_node=4, cpus_per_gpu=4)
    for i in range(7, 17):
        gpu_info['node{}'.format(i)] = dict(name='eng2080ti', gpus_per_node=4, cpus_per_gpu=4)
    for i in range(17, 22):
        gpu_info['node{}'.format(i)] = dict(name='v100', gpus_per_node=8, cpus_per_gpu=4)

    fill_content = dict(
        PARTITION=gpu_info[hostname]['name'],
        GPUS_PER_NODE=gpu_info[hostname]['gpus_per_node'],
        CPUS_PER_TASK=gpu_info[hostname]['cpus_per_gpu']
    )

    for k, v in fill_content.items():
        os.environ[k] = str(v)


def init_dog_dir():
    with open(f'setting.yml') as f:
        cfg = yaml.safe_load(f)
    dog_dirname = cfg['DOG_DIRNAME']
    os.environ['DOG_DIRNAME'] = dog_dirname

    # init dog file
    dog_dir = f"{os.environ['HOME']}/{dog_dirname}"
    if not os.path.isdir(dog_dir):
        os.makedirs(dog_dir)


init_dog_dir()
if hostname.startswith('node'):
    set_slurm_env(hostname)
    os.environ['DOG_SHELL'] = 'slurm'
else:
    os.environ['DOG_SHELL'] = 'dist'
