import os
import socket

hostname = socket.gethostname()


def set_slurm_env(hostname):
    gpu_info = dict()
    for i in range(1, 7):
        gpu_info['node{}'.format(i)] = dict(name='w2080ti', cpus_per_gpu=3)
    for i in range(7, 17):
        gpu_info['node{}'.format(i)] = dict(name='eng2080ti', cpus_per_gpu=3)
    for i in range(17, 22):
        gpu_info['node{}'.format(i)] = dict(name='v100', cpus_per_gpu=3)

    fill_content = dict(
        PARTITION=gpu_info[hostname]['name'],
        CPUS_PER_TASK=gpu_info[hostname]['cpus_per_gpu']
    )

    for k, v in fill_content.items():
        os.environ[k] = str(v)


# init watch-dog file
watch_dog_dir = f"{os.environ['HOME']}/.dog"
if not os.path.isdir(watch_dog_dir):
    os.makedirs(watch_dog_dir)

if hostname.startswith('node'):
    set_slurm_env(hostname)
    os.environ['shell'] = 'slurm'
else:
    os.environ['shell'] = 'dist'
