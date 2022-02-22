import os
import sys
import time
import socket
from termcolor import colored

import dog_api.api as api
from dog_api.utils import occupy_mem
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

    # we use config name as output-dir
    short_dir = config_path.split('/')[-1].split('.')[0]
    api.dog.summary_dog.register_task(short_dir)
    output_dir = list(output_dir.split('/'))
    output_dir[-1] = short_dir
    output_dir = '/'.join(output_dir)

    os.chdir(target_dir)
    os.system(f"bash {base_dir}/scripts/{shell}_train.sh "
              f"{config_path} {gpu_info} {output_dir}")

    api.dog.summary_dog.finish_task()


def start_tasks(info_proc, dir_and_cfgs, base_dir, target_dir, gpu_info, custom_shell):
    for (dirn, cfg) in dir_and_cfgs:
        if is_use_slurm() and gpu_info[0] == '/' and gpu_info[-1] == '/':
            gpu_info = gpu_info[1:-1].split(',')
            assert len(gpu_info) >= 2
            gpu_per_node, node_names = int(gpu_info[0]), gpu_info[1:]
            info_proc.set_slurm_env(node_names[0])
            assert gpu_per_node != 0
            gres_0_gpu = False
            if gpu_per_node < 0:
                gres_0_gpu = True
                gpu_per_node = -gpu_per_node

            os.environ['NODE_NAMES'] = ','.join(node_names)
            os.environ['NTASKS'] = str(gpu_per_node * len(node_names))
            do_job(base_dir, target_dir, cfg, dirn,
                   0 if gres_0_gpu else gpu_per_node, shell=custom_shell)
        else:
            while True:
                if socket.gethostname().startswith('gpu'):
                    thre = 0.95
                else:
                    thre = 0.8
                gpu_free_id, _ = get_free_gpu(thre=thre)
                gpu_list = gpu_info.split(',')
                if set(gpu_list).issubset(set(gpu_free_id)):
                    do_job(base_dir, target_dir, cfg, dirn, ','.join(gpu_list), shell=custom_shell)
                    time.sleep(2)
                    print("Cleaning Memory")
                    clean_memory(gpu_list)
                    break
                else:
                    wait_for = sorted(list(set(gpu_list).difference(
                        set(gpu_list).intersection(gpu_free_id))))
                    print(f"Waiting for gpus: {wait_for}\r", end="")
                time.sleep(10)
        time.sleep(5)


def clean_memory(gpu_list):
    stat_res = os.popen('${HOME}/gpustat -p').readlines()
    if isinstance(stat_res, list) and len(stat_res) > 0:
        stat_res = stat_res[1:]
        # assert len(stat_res) == gpu_num, f"{stat_res} <=> {gpu_num}"
        for gpu_id in gpu_list:
            stat_info = stat_res[int(gpu_id)]
            gpu_users_mem = stat_info.strip().split('|')[-1].split()
            for user_mem in gpu_users_mem:
                pid = user_mem.split('/')[1].split('(')[0]
                os.system(f'kill -9 {pid}')


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

    def get_custom_shell(self, target_name):
        return self.target_loader.get_custom_shell(target_name)

    def get_base_dir(self):
        return self.setting_loader.get_basedir()

    def set_target_env(self, target_name):
        os.environ['ENTRY_PATH'] = self.target_loader.get_entry_path(target_name)
        os.environ['ENTRY_KEY_DIR'] = self.target_loader.get_entry_key_dir(target_name)
        os.environ['PY_ARGS'] = self.target_loader.get_entry_extra_args(target_name)

    def set_slurm_env(self, hostname):
        slurm_env = self.master_loader.get_slurm_env(hostname)
        for k, v in slurm_env.items():
            os.environ[k] = str(v)


def main():
    args = sys.argv[1:]
    assert len(args) in [4, 5], args

    # for dist training, gpu_info: 2,3 or 0,1,2,3 or 0
    # for slurm training, gpu_info: /2, node9/ or /-4, node19, node20/
    master_name, target_name, dir_name, gpu_info = args[:4]

    info_proc = InfoProc(master_name)
    # read
    repeat_times = 1
    if len(dir_name) > 2 and dir_name[-2] == '/' and dir_name[-1].isdigit():
        repeat_times = int(dir_name[-1])
        dir_name = dir_name[:-2]
    dir_and_cfgs = info_proc.get_dir_and_cfgs(dir_name)
    dir_and_cfgs = dir_and_cfgs * repeat_times
    target_dir = info_proc.get_target_dir(target_name)
    custom_shell = info_proc.get_custom_shell(target_name)
    base_dir = info_proc.get_base_dir()
    # set
    info_proc.set_target_env(target_name)
    assert dir_and_cfgs, dir_and_cfgs

    print(f"The target name is: {target_name}, we will cd to {target_dir}")
    print("Tasks:")
    for (dirn, cfg) in dir_and_cfgs:
        print(f"  {colored(cfg, 'blue')} => {colored(dirn, 'green')} using gpu: {gpu_info}.")

    if len(args) == 5:
        wait_last_task(args[4])

    start_tasks(info_proc, dir_and_cfgs, base_dir, target_dir, gpu_info, custom_shell)


if __name__ == '__main__':
    main()
