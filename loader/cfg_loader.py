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

    def get_cfg_and_dirs(self, cfg_path):
        cfg_and_dir = []  # [[cfg1, dir1], [cfg2, dir2]]
        for root_dir in self.task_cfg:
            cfg = self.task_cfg[root_dir]
            if cfg_path in cfg:
                if not isinstance(cfg[cfg_path], str):
                    assert isinstance(cfg[cfg_path], list), cfg[cfg_path]
                    for cfg_p in cfg[cfg_path]:
                        cfg_and_dir.extend(self.get_cfg_and_dirs(cfg_p))
                else:
                    cfg_and_dir.append([cfg_path + '.py', os.path.join(
                        root_dir, cfg[cfg_path])])
        return cfg_and_dir

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
