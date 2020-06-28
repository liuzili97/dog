import time
import os
import socket
import json
import paramiko
import threading
from termcolor import colored
from tabulate import tabulate
from collections import defaultdict

from loader import MasterLoader, SettingLoader
from dog_api import get_slurm_scontrol_show, get_slurm_squeue


class Reporter():

    def __init__(self):
        self.gpus_results = {}
        self.tasks_results = {}
        self.gpus_by_us = defaultdict(list)
        self.ssh_handlers = {}

        self.disp_thre = dict(used=0.2, power=90)

        hostname = socket.gethostname()
        master_loader = MasterLoader(hostname)
        setting_loader = SettingLoader()

        self.all_cmds = dict(
            gpus='nvidia-smi --query-gpu=memory.used,memory.total,power.draw --format=csv,noheader',
            tasks=f"cat ~/{setting_loader.get_dirname()}/*"
        )
        self.use_slurm = master_loader.is_use_slurm()
        self.connect_dict = master_loader.get_connect_dict()
        self.all_nodes = master_loader.get_all_nodes()

        self.headers = self.init_headers()

    def ssh(self, name, connect_args):
        try:
            if name not in self.ssh_handlers:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(**connect_args)
                self.ssh_handlers[name] = ssh
            else:
                ssh = self.ssh_handlers[name]
            self.gpus_results[name] = ssh.exec_command(self.all_cmds['gpus'])[1].readlines()
            try:
                self.tasks_results = bytes.decode(
                    ssh.exec_command(self.all_cmds['tasks'])[1].read())
            except:
                pass
        except:
            self.gpus_results[name] = ''
        # xx.close()

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

    def single_node(self, name):
        gpus_res = self.gpus_results[name]
        gpus_info = []
        name_col = {}
        suggest_level = 0

        if isinstance(gpus_res, list) and gpus_res:
            gpu_num, free_gpu_num = len(gpus_res), 0

            for gpu_id, gpu_info in enumerate(gpus_res):
                if self.use_slurm:
                    # TODO gpu_id in self.gpus_by_us always starts from 0
                    gpu_by_us = len(self.gpus_by_us[name]) > 0
                else:
                    gpu_by_us = gpu_id in self.gpus_by_us[name]
                gpu_str, is_free = self.single_gpu(gpu_info, gpu_by_us)
                gpus_info.append(gpu_str)
                free_gpu_num += is_free

            name_col, suggest_level = self.get_name_disp_info(gpu_num, free_gpu_num)

        for _ in range(8 - len(gpus_info)):
            gpus_info.append('')

        return [colored(name, **name_col)] + gpus_info, suggest_level

    def update_gpus_by_us(self):
        for k, v in self.tasks_results.items():
            if v['eta'] != 'Done':
                gpus = v['gpus']
                for gpu in gpus:
                    hostname, device_id = gpu.split(':')
                    self.gpus_by_us[hostname].append(int(device_id))

    def update_info(self):
        threads = []
        for name, connect_args in self.connect_dict.items():
            threads.append(
                threading.Thread(target=self.ssh, args=(name, connect_args)))
        for thr in threads:
            thr.start()
        for thr in threads:
            thr.join()

        self.tasks_results = cat_tasks_str_to_dict(self.tasks_results)

        self.gpus_by_us = defaultdict(list)
        self.update_gpus_by_us()

    def init_headers(self):
        if self.use_slurm:
            gpu_headers = ['parti', 'node', 'gpu', 'cpu', 'mem'] + \
                          ['GPU{}'.format(i) for i in range(8)] + ['users']
        else:
            gpu_headers = ['node'] + ['GPU{}'.format(i) for i in range(8)]
        task_headers = ['day', 'time', 'epoch', 'it', 'eta', 'name', 'n', 'upda', 'loss', 'oth']
        return gpu_headers, task_headers

    def build_disp(self, prog_data, summary, task_data):
        prog_table = tabulate(prog_data, headers=self.headers[0], tablefmt='rst', stralign='right')
        summary_msg = colored(
            "All free ({}):\t{}".format(len(summary['node_l3']), '  '.join(summary['node_l3'])),
            'green', attrs=['bold']) + colored(
            "\nMost free ({}):\t{}".format(len(summary['node_l2']),
                                           '  '.join(summary['node_l2'])),
            'cyan') + colored(
            "\nSome free ({}):\t{}\n".format(len(summary['node_l1']),
                                             '  '.join(summary['node_l1'])),
            'yellow')
        task_table = tabulate(task_data, headers=self.headers[1], tablefmt='presto',
                              stralign='right', numalign='right')
        return prog_table, summary_msg, task_table

    def build_prog_data(self):
        prog_data = []
        summary = dict(node_l3=[], node_l2=[], node_l1=[], node_l0=[])

        for name in self.all_nodes:
            prog_info, suggest_level = self.single_node(name)
            prog_data.append(prog_info)
            summary['node_l{}'.format(suggest_level)].append(name)
        return prog_data, summary

    def build_task_data(self):
        task_data = []
        task_data_raw = defaultdict(dict)
        for k, v in self.tasks_results.items():
            month_day = int(k.split('-')[0].replace('.', ''))
            min_sec = int(k.split('-')[1].replace(':', ''))
            last_update = v.pop('last_update', '-')
            eta = v.pop('eta', 'Done')
            name = v.pop('name')
            if eta != 'Done':
                name = colored(name, 'blue')
                last_update = colored(last_update, 'cyan')
                eta = colored(eta, 'cyan')
            gpu_num = v.pop('gpu_num', '-')
            loss = colored(v.pop('loss', '-'), 'red')
            epoch_str = f"{v.pop('epoch', '-')}/{v.pop('max_epochs', '-')}"
            inner_iter = v.pop('inner_iter', '-')
            # TODO remove
            max_inner_iters = v.pop('max_inner_iter', '-')
            if max_inner_iters == '-':
                max_inner_iters = v.pop('max_inner_iters', '-')
            iter_str = '-'
            if inner_iter != '-' and max_inner_iters != '-':
                percent = int(int(inner_iter) / int(max_inner_iters) * 100)
                iter_str = f"{percent}%"
            _ = v.pop('gpus')

            task_list = [colored(min_sec, attrs=['bold']), epoch_str, iter_str,
                         eta, name, gpu_num, last_update, loss, '']
            if v:
                task_list[-1] = v.__str__().replace("'", "").replace("{", "").replace(
                    "}", "").replace(" ", "")
            task_data_raw[month_day][min_sec] = task_list

        sort_month_day = sorted(task_data_raw.keys())
        for k in sort_month_day:
            sub_task_data_raw = task_data_raw[k]
            sort_min_sec = sorted(sub_task_data_raw.keys())
            for i, k1 in enumerate(sort_min_sec):
                day = f"{k:04d}" if i == 0 else ""
                task_data.append([day] + sub_task_data_raw[k1])
        return task_data

    def add_slurm_info(self, prog_data):
        if not self.use_slurm:
            return prog_data

        new_prog_data = []
        nodes_users = get_slurm_squeue(self.all_nodes)
        for name, prog_info in zip(self.all_nodes, prog_data):
            slurm_info = get_slurm_scontrol_show(name)
            prog_info.insert(1, f"{slurm_info['alloc_mem'] / 1000:.00f}/"
                                f"{slurm_info['all_mem'] / 1000:.00f}G")
            prog_info.insert(1, f"{slurm_info['alloc_cpu']}/{slurm_info['all_cpu']}")
            name_col, _ = self.get_name_disp_info(slurm_info['all_gpu'],
                                                  slurm_info['all_gpu'] - slurm_info['alloc_gpu'])
            prog_info.insert(1, colored(f"{slurm_info['alloc_gpu']}/"
                                        f"{slurm_info['all_gpu']}", **name_col))
            prog_info.insert(0, f"{slurm_info['partitions']}")
            prog_info.append(','.join(nodes_users[name]))
            new_prog_data.append(prog_info)
        return new_prog_data

    def run(self):
        while True:
            self.update_info()

            prog_data, summary = self.build_prog_data()
            prog_data = self.add_slurm_info(prog_data)
            task_data = self.build_task_data()

            prog_table, summary_msg, task_table = self.build_disp(prog_data, summary, task_data)

            os.system('clear')
            print(prog_table)
            print(summary_msg)
            print(task_table)
            time.sleep(10)


def cat_tasks_str_to_dict(tasks_info):
    # self.tasks_results is a str at this moment
    tasks_results = {}
    if tasks_info:
        split_cat_str = tasks_info.split('}')
        split_cat_str = [s + '}' for s in split_cat_str][:-1]
        tasks_list = [json.loads(s) for s in split_cat_str]
        for task_res in tasks_list:
            key = task_res.pop('key')
            tasks_results[key] = task_res
    return tasks_results


if __name__ == '__main__':
    reporter = Reporter()
    reporter.run()
