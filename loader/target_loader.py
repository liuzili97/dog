import os
import yaml


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

    def get_target_dir(self, target_name):
        return self.target_cfgs[target_name]['DIR'][self.master_name]

    def get_entry_path(self, target_name):
        return self.target_cfgs[target_name]['ENTRY']['path']

    def get_entry_key_dir(self, target_name):
        return self.target_cfgs[target_name]['ENTRY']['key_dir']

    def get_target_dirs(self):
        return [self.target_cfgs[k]['DIR'][self.master_name]
                for k in self.target_cfgs]
