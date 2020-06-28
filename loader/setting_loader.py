import os
import yaml


def get_dog_basedir():
    paths = os.environ['PYTHONPATH'].split(':')
    dog_basedir = []
    for p in paths:
        if p.split('/')[-1] == 'dog':
            dog_basedir.append(p)
    assert len(dog_basedir) > 0, "Please add the path of 'dog' to $PYTHONPATH"
    if len(dog_basedir) > 1:
        assert len(set(dog_basedir)) == 1, f"Find multiple dogs {set(dog_basedir)}"
    return dog_basedir[0]


class SettingLoader():

    def __init__(self):
        basedir = get_dog_basedir()
        with open(f"{basedir}/cfg/setting.yml") as f:
            cfg = yaml.safe_load(f)

        self.dirname = cfg['DIRNAME']
        self.write_interval = cfg['WRITE_INTERVAL']

    def get_dirname(self):
        return self.dirname

    def get_write_interval(self):
        return self.write_interval

    def get_dir(self):
        return os.path.join(os.environ['HOME'], self.dirname)

    def get_basedir(self):
        return get_dog_basedir()
