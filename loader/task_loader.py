import os
import yaml
from collections import OrderedDict, defaultdict
from termcolor import colored

from . import is_master_right


class TaskLoader():
    """
    only init on node-host
    """
    def __init__(self, master_name):
        is_master_right(master_name)

        with open(f'task/{master_name}.yml') as f:
            cfg = yaml.safe_load(f)

        self.task_cfg = cfg['TASKS']
        self.gpu_cfg = cfg.get('DEVICES', {})
        assert self.check_no_repeat()

    def check_no_repeat(self):
        flag = True
        counter = defaultdict(int)
        for v in self.task_cfg.values():
            for k in v.keys():
                counter[k] += 1
        for k, v in counter.items():
            if v > 1:
                print(colored(f"{k} appears {v} times!"), 'red')
                flag = False
        return flag

    def get_dir_and_cfgs(self, dir_name):
        dir_and_cfg = []  # [[dir1, cfg1], [dir2, cfg2]]
        for root_dir in self.task_cfg:
            cfg = self.task_cfg[root_dir]
            if dir_name in cfg:
                if not isinstance(cfg[dir_name], str):
                    assert isinstance(cfg[dir_name], list), cfg[dir_name]
                    for dir_n in cfg[dir_name]:
                        dir_and_cfg.extend(self.get_dir_and_cfgs(dir_n))
                else:
                    dir_and_cfg.append([os.path.join(root_dir, dir_name), cfg[dir_name]])
                return dir_and_cfg

    def get_connect_dict(self):
        connect_dict = OrderedDict()
        for name, ssh_config in self.nodes_dict.items():
            connect_dict[name] = dict(
                hostname=ssh_config['ip'],
                port=ssh_config.get('port', 22),
                username=ssh_config.get('user', None),
                timeout=2
            )
        return connect_dict
