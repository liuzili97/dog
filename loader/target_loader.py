import os
import yaml

from dog_api import set_target_env


class TargetLoader():

    def __init__(self, master_name):
        # load all yml
        cfgs = {}
        for root, _, files in os.walk('cfg/target'):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    cfgs[file[:-4]] = yaml.safe_load(f)

        self.master_name = master_name
        self.target_cfgs = cfgs
        self.target_name = None

    def set_target_name(self, target_name):
        self.target_name = target_name

        path = self.target_cfgs[self.target_name]['ENTRY']['path']
        key_dir = self.target_cfgs[self.target_name]['ENTRY']['key_dir']
        set_target_env(entry_path=path, entry_key_dir=key_dir)

    def get_target_dir(self):
        assert self.target_name
        return self.target_cfgs[self.target_name]['DIR'][self.master_name]

    def get_target_dirs(self):
        return [self.target_cfgs[k]['DIR'][self.master_name]
                for k in self.target_cfgs]
