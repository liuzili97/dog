import argparse
import os
import sys
import socket
import time
from datetime import datetime
from termcolor import colored
from loader import CFGLoader
from utils import _dist_env_set, register_task, update_task_info


# from utils import get_free_gpu, get_config


def shell_name(config_path):
    keywords = ['MoCo', 'CMC', 'LinCLS']
    with open(config_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        for k in keywords:
            if k in line:
                return 'train_ssl'
    return 'train'


def do_job(target_dir, config_path, output_dir, gpu_num):
    print("Executing shell...")

    short_dir = output_dir.split('/')[-1].split('-')[0]
    register_task(short_dir)

    os.chdir(target_dir)
    sh_name = shell_name(config_path)
    os.system(
        "./scripts/{}_{}.sh {} {} {}".format(
            os.environ['shell'], sh_name,
            config_path, ','.join(list(map(str, list(range(int(gpu_num)))))), output_dir))

    update_task_info('eta', 'Done')


def is_finished(task_tuple, task_id):
    if len(task_tuple) == task_id:
        print("Works have done!")
        exit(0)


def main1():
    detected_times = 0
    task_id = 0
    first_run = True
    while True:
        if not first_run:
            time.sleep(10)

        # not need to wait in slurm
        if args.g:
            is_finished(task_tuple, task_id)

            do_job(task_tuple[task_id][0], task_tuple[task_id][1],
                   list(map(str, list(range(int(args.g))))),
                   workers_per_gpu)
            task_id += 1

            is_finished(task_tuple, task_id)
        else:
            gpu_free_id, gpu_num = get_free_gpu(memory_thre)
            remove_id = []
            for gid in gpu_free_id:
                if int(gid) not in list(watch_gpus):
                    remove_id.append(gid)
            for rid in remove_id:
                gpu_free_id.remove(rid)

            if len(gpu_free_id) > 0:
                detected_times += 1
                if len(gpu_free_id) == gpu_num:
                    info_head = "All gpus detected"
                else:
                    info_head = "Detect gpus: {}".format(gpu_free_id)
                print("{}, prepare to use, count {}.".format(
                    info_head, counter - detected_times))
            else:
                detected_times = 0
                print("No free gpu detected.")
                first_run = False
                continue

            if first_run or do_now or detected_times >= counter:
                is_finished(task_tuple, task_id)

                do_job(task_tuple[task_id][0], task_tuple[task_id][1], gpu_free_id,
                       workers_per_gpu)
                task_id += 1

                is_finished(task_tuple, task_id)
                detected_times = counter - 6
            first_run = False


def main():
    hostname = socket.gethostname()
    slurm_mode = False if hostname.startswith('fabu') else True
    loader = CFGLoader(hostname)

    args = sys.argv
    if slurm_mode:
        assert len(args) == 3, args
        slurm_main(loader, args[1:])
    else:
        assert len(args) == 1, args
        non_slurm_main(loader)


def slurm_main(loader, args):
    cfg_path, gpu_num = args
    cfg_path = cfg_path[:-3] if cfg_path.endswith('.py') else cfg_path

    cfg_and_dirs = loader.get_cfg_and_dirs(cfg_path)
    target_dir = loader.get_target_dir()
    print("Tasks:")
    for cfg_and_dir in cfg_and_dirs:
        cfg, dir = cfg_and_dir
        print(f"  {colored(cfg, 'blue')} => {colored(dir, 'green')} using {gpu_num} gpus.")

    for cfg_and_dir in cfg_and_dirs:
        cfg, dir = cfg_and_dir
        do_job(target_dir, cfg, dir, gpu_num)


def non_slurm_main(loader):
    pass


if __name__ == '__main__':
    main()
