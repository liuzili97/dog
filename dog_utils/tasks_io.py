import json
import os
import time
import socket
import torch
from datetime import datetime
import functools
import shutil


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


def load_dog_file(key):
    task_info = os.popen(f"cat {os.environ['HOME']}/.dog/{key}").read()
    return json.loads(task_info)


def get_task_eta(task_name):
    for root, _, files in os.walk(f"{os.environ['HOME']}/.dog"):
        for file in files:
            task_dict = load_dog_file(file)
            if task_dict['name'] == task_name:
                return task_dict['eta']
    return 'Done'


def update(is_read=True, is_write=True):
    def update_wrapper(old_func):
        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            if is_read:
                if 'task_dict' not in kwargs or 'key' not in kwargs:
                    key = os.environ['DOG_KEY']
                    task_dict = load_dog_file(key)
                    kwargs['task_dict'] = task_dict
                    kwargs['key'] = key

            output = old_func(*args, **kwargs)

            if is_write:
                task_dict, key = output
                task_info = json.dumps(task_dict, indent=4)
                with open(f"{os.environ['HOME']}/.dog/{key}", 'w') as f:
                    f.write(task_info)
            return output

        return new_func

    return update_wrapper


@update(is_read=False)
def register_task(task_name):
    # called by dog
    key = datetime.now().strftime('%m.%d-%H:%M')
    task_dict = dict(key=key, name=task_name, gpus=[], eta='Starting')
    os.environ['DOG_KEY'] = key
    return task_dict, key


def add_device_info():
    # called in distributed part
    hostname = socket.gethostname()
    device_id = torch.cuda.current_device()
    dir_name = f"{os.environ['HOME']}/.dog/.{os.environ['DOG_KEY']}"
    try:
        os.makedirs(dir_name)
    except:
        pass
    with open(f"{dir_name}/{hostname}:{device_id}", 'w') as f:
        f.write('')


@update()
def update_task_info(info_dict, **kwargs):
    task_dict = kwargs['task_dict']
    key = kwargs['key']

    dir_name = f"{os.environ['HOME']}/.dog/.{os.environ['DOG_KEY']}"
    if os.path.isdir(dir_name):
        for _, _, files in os.walk(dir_name):
            task_dict['gpus'] = files
        shutil.rmtree(dir_name)

    for name, value in info_dict.items():
        task_dict[name] = value
    task_dict['last_update'] = datetime.now().strftime('%H%M')

    return task_dict, key


@update(is_write=False)
def task_end(**kwargs):
    task_dict = kwargs['task_dict']
    key = kwargs['key']

    update_task_info(dict(eta='Done'), **kwargs)
    start_date = task_dict['key']
    now_date = datetime.now().strftime('%m.%d-%H:%M')

    start_stamp = time.mktime(time.strptime(start_date, '%m.%d-%H:%M'))
    now_stamp = time.mktime(time.strptime(now_date, '%m.%d-%H:%M'))

    if (now_stamp - start_stamp) / 60 < 5:
        print(f"{start_date} -- {now_date}, less than 5 min")
        os.system(f"rm {os.environ['HOME']}/.dog/{key}")
