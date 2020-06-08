import yaml
import json
import os
import socket
import torch
from datetime import datetime
import functools
from collections import OrderedDict

gpus_info = []


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


def update(is_read=True, is_write=True):
    def update_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if is_read:
                key = os.environ['WATCHDOG_KEY']
                task_info = os.popen(f"cat {os.environ['HOME']}/.dog/{key}").read()
                task_dict = json.loads(task_info)
                kwargs['task_dict'] = task_dict
                kwargs['key'] = key

            output = old_func(*args, **kwargs)
            task_dict, key = output

            if is_write:
                task_info = json.dumps(task_dict, indent=4)
                with open(f"{os.environ['HOME']}/.dog/{key}", 'w') as f:
                    f.write(task_info)
            return output
        return new_func
    return update_wrapper


@update(is_read=False)
def register_task(task_name):
    # called by watch-dog
    key = datetime.now().strftime('%m.%d-%H:%M')
    task_dict = dict(key=key, name=task_name, gpus=[], eta='Starting',
                     gpu_num='Unknow')
    os.environ['WATCHDOG_KEY'] = key
    return task_dict, key


@update(is_write=False)
def add_device_info(**kwargs):
    # called in distributed part
    global gpus_info
    task_dict = kwargs['task_dict']
    key = kwargs['key']

    hostname = socket.gethostname()
    device_id = torch.cuda.current_device()
    device_count = torch.cuda.device_count()

    gpus_info.append(f'{hostname}:{device_id}')

    if len(gpus_info) == device_count:
        task_dict['gpus'] = gpus_info
        task_dict['gpu_num'] = device_count
        gpus_info = []
        task_info = json.dumps(task_dict, indent=4)
        with open(f"{os.environ['HOME']}/.dog/{key}", 'w') as f:
            f.write(task_info)


@update()
def update_task_info(name, value, **kwargs):
    task_dict = kwargs['task_dict']
    key = kwargs['key']

    task_dict[name] = value

    return task_dict, key

