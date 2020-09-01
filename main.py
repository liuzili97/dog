import os
import sys
import time
from termcolor import colored

import dog_api.api as api
from loader import MasterLoader, TaskLoader, TargetLoader, SettingLoader
from dog_api import get_free_gpu, is_use_slurm


# TODO remove
def shell_name(config_path):
    keywords = ['MoCo', 'CMC', 'LinCLS']
    with open(config_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        for k in keywords:
            if k in line:
                return 'train_ssl'
    return 'train'


def do_job(base_dir, target_dir, config_path, output_dir, gpu_info, shell='slurm'):
    print("Executing shell...")

    short_dir = output_dir.split('/')[-1].split('-')[0]
    api.dog.summary_dog.register_task(short_dir)

    os.chdir(target_dir)
    os.system(f"bash {base_dir}/scripts/{shell}_train.sh "
              f"{config_path} {gpu_info} {output_dir}")

    api.dog.summary_dog.finish_task()


def start_tasks(dir_and_cfgs, base_dir, target_dir, gpu_info):
    for (dirn, cfg) in dir_and_cfgs:
        if is_use_slurm():
            do_job(base_dir, target_dir, cfg, dirn, gpu_info, shell='slurm')
        else:
            while True:
                gpu_free_id, _ = get_free_gpu(thre=0.9)
                gpu_list = gpu_info.split(',')
                if set(gpu_list).issubset(set(gpu_free_id)):
                    do_job(base_dir, target_dir, cfg, dirn, ','.join(gpu_list), shell='dist')
                    break
                else:
                    wait_for = sorted(list(set(gpu_list).difference(
                        set(gpu_list).intersection(gpu_free_id))))
                    print(f"Waiting for gpus: {wait_for}")
                time.sleep(10)
        time.sleep(5)


def wait_last_task(last_target_name):
    while True:
        try:
            eta = api.dog.summary_dog.get_task_eta(last_target_name)
            if eta == 'Done':
                break
            print(f"Waiting for {last_target_name} to finish, eta: {eta}.")
        except:
            pass
        time.sleep(30)


class InfoProc():

    def __init__(self, master_name):
        self.master_loader = MasterLoader(master_name)
        self.task_loader = TaskLoader(master_name)
        self.target_loader = TargetLoader(master_name)
        self.setting_loader = SettingLoader()

    def get_dir_and_cfgs(self, dir_name):
        return self.task_loader.get_dir_and_cfgs(dir_name)

    def get_target_dir(self, target_name):
        return self.target_loader.get_target_dir(target_name)

    def get_base_dir(self):
        return self.setting_loader.get_basedir()

    def set_target_env(self, target_name):
        os.environ['ENTRY_PATH'] = self.target_loader.get_entry_path(target_name)
        os.environ['ENTRY_KEY_DIR'] = self.target_loader.get_entry_key_dir(target_name)
        os.environ['PY_ARGS'] = self.target_loader.get_entry_extra_args(target_name)

    def set_slurm_env(self):
        slurm_env = self.master_loader.get_slurm_env()
        for k, v in slurm_env.items():
            os.environ[k] = str(v)


def main():
    args = sys.argv[1:]
    assert len(args) in [4, 5], args

    master_name, target_name, dir_name, gpu_info = args[:4]

    info_proc = InfoProc(master_name)
    # read
    dir_and_cfgs = info_proc.get_dir_and_cfgs(dir_name)
    target_dir = info_proc.get_target_dir(target_name)
    base_dir = info_proc.get_base_dir()
    # set
    info_proc.set_target_env(target_name)
    if is_use_slurm():
        info_proc.set_slurm_env()
    assert dir_and_cfgs, dir_and_cfgs

    print(f"The target name is: {target_name}, we will cd to {target_dir}")
    print("Tasks:")
    for (dirn, cfg) in dir_and_cfgs:
        print(f"  {colored(cfg, 'blue')} => {colored(dirn, 'green')} using {gpu_info} gpus.")

    if len(args) == 5:
        wait_last_task(args[4])

    start_tasks(dir_and_cfgs, base_dir, target_dir, gpu_info)


if __name__ == '__main__':
    main()
