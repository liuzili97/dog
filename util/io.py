import os
from datetime import datetime

from dog_api import load_dog_file, update, set_dog_key, set_gpu_num


def get_task_eta(task_name):
    for root, _, files in os.walk(f"{os.environ['DOG_DIR']}"):
        for file in files:
            task_dict = load_dog_file(file)
            if task_dict['name'] == task_name:
                return task_dict['eta']
    return 'Done'


@update(is_read=False)
def register_task(task_name, gpu_info):
    # called by dog
    gpu_num = len(gpu_info) if isinstance(gpu_info, list) else gpu_info
    set_gpu_num(gpu_num)
    key = datetime.now().strftime('%m.%d-%H:%M')
    task_dict = dict(key=key, name=task_name, eta='Starting', gpu_num=gpu_num, gpus=[])
    set_dog_key(key)
    return task_dict, key
