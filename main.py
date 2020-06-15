import os
import sys
import socket
import time
from termcolor import colored
from loader import CFGLoader
from dog_utils import _dist_env_set, register_task, task_end, get_free_gpu, get_task_eta


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


def start_tasks(dir_and_cfgs, slurm_mode, target_dir, gpu_info):
    for (dirn, cfg) in dir_and_cfgs:
        if slurm_mode:
            do_job(target_dir, cfg, dirn, gpu_info)
        else:
            while True:
                gpu_free_id, _ = get_free_gpu(thre=0.9)
                gpu_list = gpu_info.split(',')
                if set(gpu_list).issubset(set(gpu_free_id)):
                    do_job(target_dir, cfg, dirn, gpu_list)
                else:
                    wait_for = sorted(list(set(gpu_list).difference(
                        set(gpu_list).intersection(gpu_free_id))))
                    print(f"Waiting for gpus: {wait_for}")
                time.sleep(10)
        time.sleep(5)


def wait_last_task(last_target_name):
    while True:
        try:
            eta = get_task_eta(last_target_name)
            if eta == 'Done':
                break
            print(f"Waiting for {last_target_name} to end, eta: {eta}.")
        except:
            pass
        time.sleep(30)


def main():
    hostname = socket.gethostname()
    loader = CFGLoader(hostname)

    args = sys.argv[1:]
    assert len(args) in [3, 4], args

    target_name, dir_name, gpu_info = args[:3]
    dir_and_cfgs = loader.get_dir_and_cfgs(dir_name)
    target_dir = loader.get_target_dir(target_name)
    slurm_mode = loader.is_use_slurm()
    assert dir_and_cfgs, dir_and_cfgs

    print(f"The target name is: {target_name}, we will cd to {target_dir}")
    print("Tasks:")
    for (dirn, cfg) in dir_and_cfgs:
        print(f"  {colored(cfg, 'blue')} => {colored(dirn, 'green')} using {gpu_info} gpus.")

    if len(args) == 4:
        wait_last_task(args[3])

    start_tasks(dir_and_cfgs, slurm_mode, target_dir, gpu_info)


if __name__ == '__main__':
    main()
