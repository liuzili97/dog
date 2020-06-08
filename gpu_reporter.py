import time
import json
import os
import socket
import paramiko
import threading
from termcolor import colored
from tabulate import tabulate
from collections import defaultdict
from loader import RemoteLoader
from utils import cat_tasks_str_to_dict


class Reporter():

    def __init__(self):
        self.gpus_results = {}
        self.tasks_results = {}
        self.gpus_by_us = defaultdict(list)
        self.ssh_handlers = {}

        self.disp_thre = dict(used=0.2, power=60)
        self.all_cmds = dict(
            gpus='nvidia-smi --query-gpu=memory.used,memory.total,power.draw --format=csv,noheader',
            tasks='cat ~/.dog/*'
        )

        hostname = socket.gethostname()
        loader = RemoteLoader(hostname)
        self.connect_dict = loader.get_connect_dict()
        self.all_nodes = loader.get_all_nodes()

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
        percent = int(memory_used * 10)
        gpu_str = colored(mark, prog_col, attrs=['bold']) + \
                  colored(mark * percent + ' ' * (9 - percent), prog_col) + \
                  colored('|', attrs=['dark'])
        return gpu_str, is_free

    def single_node(self, name):
        gpus_res = self.gpus_results[name]
        gpus_info = []
        name_col = {}
        suggest_level = 0

        if isinstance(gpus_res, list) and gpus_res:
            gpu_num, free_gpu_num = len(gpus_res), 0

            for gpu_id, gpu_info in enumerate(gpus_res):
                gpu_by_us = gpu_id in self.gpus_by_us[name]
                gpu_str, is_free = self.single_gpu(gpu_info, gpu_by_us)
                gpus_info.append(gpu_str)
                free_gpu_num += is_free

            name_col, suggest_level = self.get_name_disp_info(gpu_num, free_gpu_num)

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
        gpu_headers = ['node'] + ['GPU{}'.format(i) for i in range(8)]
        task_headers = ['day', 'time', 'epoch', 'it', 'eta', 'name', 'n', 'upda', 'loss', 'oth']
        return gpu_headers, task_headers

    def build_disp(self, prog_data, summary, task_data):
        prog_table = tabulate(prog_data, headers=self.headers[0], tablefmt='rst')
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
            eta = colored(eta, 'cyan')
            gpu_num = colored(v.pop('gpu_num', '-'), 'cyan')
            loss = colored(v.pop('loss', '-'), 'red')
            epoch_str = f"{v.pop('epoch', '-')}/{v.pop('max_epochs', '-')}"
            inner_iter = v.pop('inner_iter', '-')
            max_inner_iter = v.pop('max_inner_iter', '-')
            iter_str = '-'
            if inner_iter != '-' and max_inner_iter != '-':
                percent = int(int(inner_iter) / int(max_inner_iter) * 100)
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

    def run(self):
        while True:
            self.update_info()

            prog_data, summary = self.build_prog_data()
            task_data = self.build_task_data()

            prog_table, summary_msg, task_table = self.build_disp(prog_data, summary, task_data)

            os.system('clear')
            print(prog_table)
            print(summary_msg)
            print(task_table)
            time.sleep(10)


if __name__ == '__main__':
    reporter = Reporter()
    reporter.run()
