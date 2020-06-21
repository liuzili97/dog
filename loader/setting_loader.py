import os
import yaml


class SettingLoader():

    def __init__(self):
        with open(f"{os.environ['DOG_BASEDIR']}/cfg/setting.yml") as f:
            cfg = yaml.safe_load(f)

        self.dirname = cfg['DIRNAME']
        self.write_interval = cfg['WRITE_INTERVAL']

    def get_dirname(self):
        return self.dirname

    def get_write_interval(self):
        return self.write_interval

    def get_dir(self):
        return os.path.join(os.environ['HOME'], self.dirname)
