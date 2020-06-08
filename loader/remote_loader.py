import yaml
from collections import OrderedDict


class RemoteLoader():

    def __init__(self, hostname):
        with open(f'remote/{hostname}.yml') as f:
            gpus_cfg = yaml.safe_load(f)

        self.nodes_dict = gpus_cfg['NODES']

    def get_all_nodes(self):
        return list(self.nodes_dict.keys())

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
