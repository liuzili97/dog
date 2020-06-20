import os
import json
import functools
import time
import socket
from datetime import datetime
import shutil
from termcolor import colored

from .dist_util import master_only
from .env import dog_launched


def load_dog_file(key):
    task_info = os.popen(f"cat {os.environ['DOG_DIR']}/{key}").read()
    return json.loads(task_info)


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
                with open(f"{os.environ['DOG_FILE']}", 'w') as f:
                    f.write(task_info)
            return output

        return new_func

    return update_wrapper


def add_device_info():
    # called in distributed part
    if 'DOG_DEVICEDIR' not in os.environ:
        print(colored("dog is not launched, cannot find env 'DOG_DEVICEDIR'", 'red'))
        return
    import torch
    device_id = torch.cuda.current_device()
    dir_name = os.environ['DOG_DEVICEDIR']
    try:
        os.makedirs(dir_name)
    except:
        pass
    with open(f"{dir_name}/{socket.gethostname()}:{device_id}", 'w') as f:
        f.write('')


@dog_launched
@master_only
@update()
def update_task_info(info_dict, **kwargs):
    task_dict = kwargs['task_dict']
    key = kwargs['key']

    dir_name = os.environ['DOG_DEVICEDIR']
    if os.path.isdir(dir_name):
        for _, _, files in os.walk(dir_name):
            task_dict['gpus'] = files
        shutil.rmtree(dir_name)

    for name, value in info_dict.items():
        task_dict[name] = value
    task_dict['last_update'] = datetime.now().strftime('%d-%H%M')

    return task_dict, key


@update(is_write=False)
def task_end(**kwargs):
    task_dict = kwargs['task_dict']

    update_task_info(dict(eta='Done'), **kwargs)
    start_date = task_dict['key']
    now_date = datetime.now().strftime('%m.%d-%H:%M')

    start_stamp = time.mktime(time.strptime(start_date, '%m.%d-%H:%M'))
    now_stamp = time.mktime(time.strptime(now_date, '%m.%d-%H:%M'))

    if (now_stamp - start_stamp) / 60 < 5:
        print(f"{start_date} -- {now_date}, less than 5 min")
        os.system(f"rm {os.environ['DOG_FILE']}")
