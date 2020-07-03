import time
import os
import socket
import paramiko
import threading

from monitor import GPUMonitor, TaskMonitor
from loader import MasterLoader


class Monitor():

    def __init__(self):
        self.ssh_handlers = {}

        master_loader = MasterLoader(socket.gethostname())
        self.connect_dict = master_loader.get_connect_dict()
        self.all_nodes = master_loader.get_all_nodes()

        self.monitors = [GPUMonitor(), TaskMonitor()]

    def ssh_func(self, name, connect_args):
        is_first_thread = name == self.all_nodes[0]
        if name not in self.ssh_handlers:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(**connect_args)
                self.ssh_handlers[name] = ssh
            except:
                self.ssh_handlers[name] = None
        else:
            ssh = self.ssh_handlers[name]

        for m in self.monitors:
            m.update_info(name, ssh, is_first_thread)

    def update_info(self):
        threads = []
        for name, connect_args in self.connect_dict.items():
            threads.append(
                threading.Thread(target=self.ssh_func, args=(name, connect_args)))
        for thr in threads:
            thr.start()
        for thr in threads:
            thr.join()

    def run(self):
        while True:
            self.update_info()

            print_res = []
            for m in self.monitors:
                print_res.extend(m.build_disp(self.all_nodes))

            os.system('clear')
            for res in print_res:
                print(res)
            time.sleep(10)


if __name__ == '__main__':
    monitor = Monitor()
    monitor.run()
