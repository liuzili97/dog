import os
import re
import time
from termcolor import colored
from tabulate import tabulate
from collections import defaultdict

from .base import BaseMonitor
from loader import SettingLoader
from dog_api import get_slurm_scontrol_show, get_slurm_squeue, is_use_slurm


class SLURMMonitor(BaseMonitor):

    def __init__(self):
        self.gpus_results, self.stat_results = {}, {}
        self.nodes_users_gpus = dict()

        setting_loader = SettingLoader()
        self.report_dirname = setting_loader.get_report_dirname()
        self.headers = self.init_headers()

    def init_headers(self):
        return ['parti', 'node', 'gpu', 'cpu', 'mem', 'users']

    def update_info(self, name, ssh, is_first_thread):
        pass

    def build_data(self, all_nodes):
        prog_data = []

        for name in all_nodes:
            prog_info = self.single_node(name)
            prog_data.append(prog_info)

        prog_data = self.add_slurm_info(all_nodes, prog_data)
        return prog_data

    def is_valid(self, res):
        return isinstance(res, list) and len(res) > 0

    def single_node(self, name):
        return [name]

    def get_name_disp_info(self, gpu_num, free_gpu_num):
        suggest_level = 0
        if 0 < gpu_num == free_gpu_num:
            name_col = dict(color='green', attrs=['bold'])
            suggest_level = 3
        elif 0 < free_gpu_num < gpu_num:
            if free_gpu_num / gpu_num >= 0.5:
                name_col = dict(color='cyan', attrs=[])
                suggest_level = 2
            else:
                name_col = dict(color='yellow', attrs=[])
                suggest_level = 1
        elif gpu_num == 0:
            name_col = dict(color='white', attrs=['dark'])
        else:
            name_col = dict(color='white', attrs=[])
        return name_col, suggest_level

    def add_slurm_info(self, all_nodes, prog_data):
        new_prog_data = []
        nodes_users = get_slurm_squeue(all_nodes)
        for name, prog_info in zip(all_nodes, prog_data):
            slurm_info = get_slurm_scontrol_show(name)
            prog_info.insert(1, f"{slurm_info['alloc_mem'] / 1000:.00f}/"
                                f"{slurm_info['all_mem'] / 1000:.00f}G")
            prog_info.insert(1, f"{slurm_info['alloc_cpu']}/{slurm_info['all_cpu']}")
            name_col, _ = self.get_name_disp_info(slurm_info['all_gpu'],
                                                  slurm_info['all_gpu'] - slurm_info['alloc_gpu'])
            prog_info.insert(1, colored(f"{slurm_info['alloc_gpu']}/"
                                        f"{slurm_info['all_gpu']}", **name_col))
            prog_info.insert(0, f"{slurm_info['partitions']}")
            node_users = nodes_users[name]
            node_users = self.add_user_gpus(node_users, self.nodes_users_gpus.get(name, None),
                                            all_gpu_num=slurm_info['all_gpu'])
            node_users = self.add_bold_for_me(node_users)

            prog_info.append(colored('| ', attrs=['bold']).join(node_users))
            new_prog_data.append(prog_info)
        return new_prog_data

    def add_user_gpus(self, node_users, users_gpus, all_gpu_num=None):

        def _add_gpu_list(user, gpu_li):
            if isinstance(all_gpu_num, int) and len(set(gpu_li)) == all_gpu_num:
                return user + "(all)"
            return user + f"({','.join(gpu_li)})"

        new_node_users = []
        for user in node_users:
            if users_gpus:
                for full_user, gpu_list in users_gpus.items():
                    if full_user.startswith(user):
                        user = _add_gpu_list(user, gpu_list)
                        break
            new_node_users.append(user)

        if users_gpus:
            for full_user, gpu_list in users_gpus.items():
                match = False
                for user in node_users:
                    if full_user.startswith(user):
                        match = True
                        break
                if not match:
                    full_user = _add_gpu_list(full_user, gpu_list)
                    col = 'red' if is_use_slurm() else 'white'
                    new_node_users.append(colored(full_user, col))

        return new_node_users

    def add_bold_for_me(self, users):
        new_users = []
        users = sorted(list(users))
        for i, u in enumerate(users):
            if os.environ['USER'].startswith(u.split('(')[0]):
                u = colored(u, attrs=['bold'])
            new_users.append(u)
        return new_users

    def build_disp(self, all_nodes):
        prog_data = self.build_data(all_nodes)
        prog_table = tabulate(prog_data, headers=self.headers, tablefmt='rst', stralign='right')
        return prog_table, ''
