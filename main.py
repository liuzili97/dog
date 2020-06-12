import os
import sys
import socket
import time
from termcolor import colored
from loader import CFGLoader
from dog_utils import _dist_env_set, register_task, task_end, get_free_gpu


def shell_name(config_path):
    keywords = ['MoCo', 'CMC', 'LinCLS']
    with open(config_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        for k in keywords:
            if k in line:
                return 'train_ssl'
    return 'train'


def do_job(target_dir, config_path, output_dir, gpu_info):
    print("Executing shell...")

    short_dir = output_dir.split('/')[-1].split('-')[0]
    register_task(short_dir)

    os.chdir(target_dir)
    sh_name = shell_name(config_path)
    os.system(
        "./scripts/{}_{}.sh {} {} {}".format(
            os.environ['DOG_SHELL'], sh_name,
            config_path, gpu_info, output_dir))

    task_end()


def main():
    hostname = socket.gethostname()
    loader = CFGLoader(hostname)

    args = sys.argv[1:]
    assert len(args) == 3, args

    target_name, dir_name, gpu_info = args
    dir_and_cfgs = loader.get_dir_and_cfgs(dir_name)
    target_dir = loader.get_target_dir(target_name)
    slurm_mode = loader.is_use_slurm()
    assert dir_and_cfgs, dir_and_cfgs

    print(f"The target name is: {target_name}, we will cd to {target_dir}")
    print("Tasks:")
    gpu_num = gpu_info if slurm_mode else len(gpu_info.split(','))
    for (dirn, cfg) in dir_and_cfgs:
        print(f"  {colored(cfg, 'blue')} => {colored(dirn, 'green')} using {gpu_num} gpus.")

    for (dirn, cfg) in dir_and_cfgs:
        if slurm_mode:
            do_job(target_dir, cfg, dirn, gpu_num)
        else:
            while True:
                gpu_free_id, _ = get_free_gpu(thre=0.9)
                gpu_list = gpu_info.split(',')
                if set(gpu_list).issubset(set(gpu_free_id)):
                    do_job(target_dir, cfg, dirn, gpu_num)
                else:
                    wait_for = sorted(list(set(gpu_list).difference(
                        set(gpu_list).intersection(gpu_free_id))))
                    print(f"Waiting for gpus: {wait_for}")
                time.sleep(10)

        time.sleep(5)


if __name__ == '__main__':
    main()
