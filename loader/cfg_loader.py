import os
import yaml
from collections import OrderedDict


class CFGLoader():

    def __init__(self, hostname):
        hostname_prefix = hostname[:4]
        with open(f'cfg/{hostname_prefix}.yml') as f:
            cfg = yaml.safe_load(f)

        self.target = cfg['TARGET']
        self.task_cfg = cfg['TASKS']
        self.gpu_cfg = cfg.get('DEVICES', {})

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

    def get_target_dir(self):
        return self.target

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
