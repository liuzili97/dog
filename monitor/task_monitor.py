import os
import json
from termcolor import colored
from tabulate import tabulate
from collections import defaultdict

from .base import BaseMonitor
from loader import SettingLoader


class TaskMonitor(BaseMonitor):

    def __init__(self):
        setting_loader = SettingLoader()

        self.tasks_results = {}
        self.cmds = dict(
            tasks=f"cat ~/{setting_loader.get_dirname()}/*"
        )
        self.headers = self.init_headers()

    def init_headers(self):
        return ['day', 'time', 'epoch', 'it', 'eta', 'name', 'n', 'upda', 'loss', 'oth']

    def update_info(self, name, func, is_first_node):
        if not is_first_node:
            return
        tasks_info = os.popen(self.cmds['tasks']).read()
        self.tasks_results = {}
        if tasks_info:
            split_cat_str = tasks_info.split('}')
            split_cat_str = [s + '}' for s in split_cat_str][:-1]
            tasks_list = [json.loads(s) for s in split_cat_str]
            for task_res in tasks_list:
                key = task_res.pop('key')
                self.tasks_results[key] = task_res

    def build_data(self):
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
            if 'gpus' in v:
                _ = v.pop('gpus')  # TODO remove

            task_list = [colored(f"{min_sec:04d}", attrs=['bold']), epoch_str, iter_str,
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

    def build_disp(self, all_nodes):
        task_data = self.build_data()
        task_table = tabulate(task_data, headers=self.headers, tablefmt='presto',
                              stralign='right', numalign='right')
        return [task_table]
