import os
import yaml
import socket
from collections import OrderedDict

from dog_api import set_dog_shell, set_env


class MasterLoader():

    def __init__(self, master_name):
        with open(f'cfg/master/{master_name}.yml') as f:
            master_cfg = yaml.safe_load(f)

        self.use_slurm = master_cfg['USE_SLURM']
        self.nodes_dict = master_cfg['NODES']

    def get_all_nodes(self):
        return list(self.nodes_dict.keys())

    def is_use_slurm(self):
        assert isinstance(self.use_slurm, bool)
        # TODO trick
        set_dog_shell(self.use_slurm)
        return self.use_slurm

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

    def set_slurm_env(self):
        assert self.use_slurm
        node_name = socket.gethostname()
        assert node_name in self.get_all_nodes()

        node_dict = self.nodes_dict[node_name]
        env_setting = dict(
            PARTITION=node_dict['partition'],
            GPUS_PER_NODE=node_dict['gpus_per_node'],
            CPUS_PER_TASK=node_dict['cpus_per_gpu'])

        set_env(env_setting)


def is_master_right(master_name):
    master_loader = MasterLoader(master_name)
    hostname = socket.gethostname()
    all_nodes = master_loader.get_all_nodes()
    assert hostname in all_nodes, f"{hostname} not in {all_nodes}."
