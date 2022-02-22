import time
import os
import sys
import threading

from monitor import GPUMonitor, TaskMonitor
from loader import MasterLoader


class MonitorRep():

    def __init__(self, monitors):
        self.ssh_handlers = {}

        master_loader = MasterLoader('tc6000')
        self.all_nodes = master_loader.get_all_nodes()

        self.monitors = monitors

    def update_func(self, name):
        is_first_thread = name == self.all_nodes[0]

        for m in self.monitors:
            m.update_info(name, None, is_first_thread)

    def update_info(self):
        threads = []
        for name in self.all_nodes:
            threads.append(
                threading.Thread(target=self.update_func, args=(name,)))
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
    inp = sys.argv[1:]
    if len(inp) == 1:
        if inp[0] == 'g':
            monitors = [GPUMonitor()]
        elif inp[0] == 't':
            monitors = [TaskMonitor()]
        else:
            raise NotImplementedError
    else:
        monitors = [GPUMonitor(), TaskMonitor()]

    monitor = MonitorRep(monitors)
    monitor.run()
