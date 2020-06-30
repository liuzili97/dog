import os
from termcolor import colored
from tabulate import tabulate
from collections import defaultdict

from .base import BaseMonitor
from dog_api import get_slurm_scontrol_show, get_slurm_squeue, is_use_slurm


class GPUMonitor(BaseMonitor):

    def __init__(self):
        self.gpus_results, self.stat_results = {}, {}
        self.nodes_users_gpus = dict()

        self.disp_thre = dict(used=0.2, power=90)
        self.cmds = dict(
            smi='nvidia-smi --query-gpu=memory.used,memory.total,power.draw --format=csv,noheader',
            stat='${HOME}/gpustat'
        )

        self.headers = self.init_headers()

    def init_headers(self):
        if is_use_slurm():
            return ['parti', 'node', 'gpu', 'cpu', 'mem'] + \
                   ['GPU{}'.format(i) for i in range(8)] + ['users']
        return ['node'] + ['GPU{}'.format(i) for i in range(8)]

    def update_info(self, name, ssh, is_first_node):
        try:
            self.gpus_results[name] = ssh.exec_command(self.cmds['smi'])[1].readlines()
        except:
            self.gpus_results[name] = ''
        try:
            self.stat_results[name] = ssh.exec_command(self.cmds['stat'])[1].readlines()
        except:
            self.stat_results[name] = ''

    def build_data(self, all_nodes):
        prog_data = []
        summary = dict(node_l3=[], node_l2=[], node_l1=[], node_l0=[])

        for name in all_nodes:
            prog_info, suggest_level = self.single_node(name)
            prog_data.append(prog_info)
            summary['node_l{}'.format(suggest_level)].append(name)

        if is_use_slurm():
            prog_data = self.add_slurm_info(all_nodes, prog_data)
        return prog_data, summary

    def is_valid(self, res):
        return isinstance(res, list) and len(res) > 0

    def single_node(self, name):
        gpus_res = self.gpus_results[name]
        stat_res = self.stat_results[name]
        gpus_info = []
        name_col = {}
        suggest_level = 0

        if self.is_valid(gpus_res):
            gpu_num, free_gpu_num = len(gpus_res), 0
            gpus_by_us = []

            if self.is_valid(stat_res):
                stat_res = stat_res[1:]
                assert len(stat_res) == gpu_num, f"{stat_res} <=> {gpu_num}"
                users_gpus = defaultdict(list)
                for gpu_id, stat_info in enumerate(stat_res):
                    if os.environ['USER'] in stat_info:
                        gpus_by_us.append(gpu_id)
                    gpu_users_mem = stat_info.strip().split('|')[-1].split()
                    for user_mem in gpu_users_mem:
                        user = user_mem.split('(')[0]
                        users_gpus[user].append(str(gpu_id))
                self.nodes_users_gpus[name] = users_gpus

            for gpu_id, gpu_info in enumerate(gpus_res):
                gpu_by_us = gpu_id in gpus_by_us
                gpu_str, is_free = self.single_gpu(gpu_info, gpu_by_us)
                gpus_info.append(gpu_str)
                free_gpu_num += is_free

            name_col, suggest_level = self.get_name_disp_info(gpu_num, free_gpu_num)

        for _ in range(8 - len(gpus_info)):
            gpus_info.append('')

        return [colored(name, **name_col)] + gpus_info, suggest_level

    def single_gpu(self, gpu_info, gpu_by_us):
        # get ['449', '32510', '55.05'] from '449 MiB, 32510 MiB, 55.05 W'
        gpu_info = [item.split()[0] for item in gpu_info.split(',')]
        for i, item in enumerate(gpu_info):
            # return [Not Supported] for some gpus
            gpu_info[i] = float(item) if 'No' not in item else 0.

        memory_used = gpu_info[0] / gpu_info[1]
        power = gpu_info[2]
        prog_col, is_free = self.get_prog_disp_info(memory_used, power)

        mark = '|' if not gpu_by_us else '>'
        max_prog_num = 8
        percent = int(memory_used * max_prog_num)
        gpu_str = colored(mark, prog_col, attrs=['bold']) + \
                  colored(mark * percent + ' ' * (max_prog_num - percent - 2), prog_col)
        if percent < max_prog_num - 1:
            gpu_str += colored('|', attrs=['dark'])
        return gpu_str, is_free

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

    def get_prog_disp_info(self, memory_used, power):
        much_used = memory_used > self.disp_thre['used']
        much_power = power > self.disp_thre['power']
        is_free = 0
        if not much_used and not much_power:
            prog_col = 'green'
            is_free = 1
        elif much_used:
            prog_col = 'red'
        elif much_power:
            prog_col = 'yellow'
        return prog_col, is_free

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

    def add_user_gpus(self, node_users, users_gpus, all_gpu_num):
        new_node_users = []
        for user in node_users:
            if users_gpus:
                for full_user, gpu_list in users_gpus.items():
                    if full_user.startswith(user):
                        if len(set(gpu_list)) == all_gpu_num:
                            user += "(all)"
                        else:
                            user += f"({','.join(gpu_list)})"
            new_node_users.append(user)
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
        prog_data, summary = self.build_data(all_nodes)
        prog_table = tabulate(prog_data, headers=self.headers, tablefmt='rst', stralign='right')
        summary_msg = colored(
            "All free ({}):\t{}".format(len(summary['node_l3']), '  '.join(summary['node_l3'])),
            'green', attrs=['bold']) + colored(
            "\nMost free ({}):\t{}".format(len(summary['node_l2']),
                                           '  '.join(summary['node_l2'])),
            'cyan') + colored(
            "\nSome free ({}):\t{}\n".format(len(summary['node_l1']),
                                             '  '.join(summary['node_l1'])),
            'yellow')
        return prog_table, summary_msg
