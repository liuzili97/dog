import time
import os
import socket
import paramiko
import threading

from monitor import GPUMonitor, TaskMonitor
from loader import SettingLoader


class Reporter():

    def __init__(self):
        self.hostname = socket.gethostname()

        self.setting_loader = SettingLoader()
        self.report_dirname = self.setting_loader.get_report_dirname()

        self.cmds = dict(
            smi='nvidia-smi --query-gpu=memory.used,memory.total,power.draw --format=csv,noheader',
            stat='${HOME}/gpustat'
        )

    def run(self):
        while True:
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n'
            smi_res = os.popen(self.cmds['smi']).read()
            stat_res = os.popen(self.cmds['stat']).read()

            info = [time_str + '\n', smi_res + '\n', stat_res]

            with open(os.path.join(self.report_dirname, self.hostname), 'w') as f:
                f.writelines(info)

            time.sleep(10)


if __name__ == '__main__':
    monitor = Reporter()
    monitor.run()
