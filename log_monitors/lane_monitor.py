import os
import socks
import socket
import time
import telepot
import numpy as np

from tabulate import tabulate
from collections import defaultdict
from termcolor import colored

socks.set_default_proxy(socks.SOCKS5, "localhost", 7891)
socket.socket = socks.socksocket


class Monitor():

    def __init__(self, path, recent_days=60, follow_links=False, max_epoch=11, wait_min=180):
        self.path = path
        self.recent_days = recent_days
        self.follow_links = follow_links
        self.max_epoch = max_epoch
        self.wait_min = wait_min

        self.bot = telepot.Bot('2145587920:AAH8zYUkW5iWAX5ZK6xuQtbamB71qTjnxdA')
        self.valid_chat_id = [1818849255]

        self.tele_sent = []

    def search_log(self):
        def taketime(elem):
            return elem[0]

        log_files = []
        for root, _, files in os.walk(self.path, followlinks=self.follow_links):
            for file in files:
                if file.endswith('eval.log'):
                    time_pass = time.time() - time.mktime(
                        time.strptime(root.split('/')[-1], "%Y%m%d-%H%M%S"))
                    if time_pass < 3600 * 24 * self.recent_days:
                        log_files.append((time_pass, os.path.join(root, file)))

        log_files.sort(key=taketime, reverse=True)
        log_files = [log_file[1] for log_file in log_files]
        return log_files

    def send_tele(self, table_line):
        exp_name = table_line[1]
        if exp_name not in self.tele_sent:
            self.bot.sendMessage(
                self.valid_chat_id[0],
                f"{exp_name}: {table_line[2]} {table_line[3]}")
            self.tele_sent.append(exp_name)
            if len(self.tele_sent) > 20:
                self.tele_sent.pop(0)

    def stat_logs(self):
        log_files = self.search_log()

        table = []
        data_raw = defaultdict(list)
        for log_file in log_files:
            try:
                lines = open(log_file, 'r').readlines()

                metrics = defaultdict(list)
                if len(lines) == 0:
                    continue
                for i, line in enumerate(lines):
                    splitted_line = line.strip().split(' ')
                    metrics['epoch'].append(int(splitted_line[0]) + 1)
                    metrics['acc'].append(splitted_line[1])
                    metrics['miou'].append(splitted_line[2])
                    metrics['mrec'].append(splitted_line[3])
                    metrics['mprec'].append(splitted_line[4])
                    metrics['mf1'].append(splitted_line[5])

                metrics['acc'] = np.array(metrics['acc']).astype(np.float32) * 100
                metrics['miou'] = np.array(metrics['miou']).astype(np.float32) * 100
                metrics['mrec'] = np.array(metrics['mrec']).astype(np.float32) * 100
                metrics['mprec'] = np.array(metrics['mprec']).astype(np.float32) * 100
                metrics['mf1'] = np.array(metrics['mf1']).astype(np.float32) * 100

                max_miou = metrics['miou'].max()
                max_mf1 = metrics['mf1'].max()
                max_miou_str = f"{max_miou:5.2f} ({metrics['epoch'][metrics['miou'].argmax()]:>2d})"
                max_mf1_str = f"{max_mf1:5.2f} ({metrics['epoch'][metrics['mf1'].argmax()]:>2d})"

                task_name = f'{log_file.split("/")[-3]}'
                time_str = time.strftime('%m%d-%H%M',
                                         time.localtime(time.mktime(time.strptime(
                                             log_file.split('/')[-2], "%Y%m%d-%H%M%S"))))
                month_day = time_str.split('-')[0]
                hour_min_str = time_str.split('-')[1]

                time_modify = os.path.getmtime(log_file)
                time_modify_str = time.strftime('%d-%H%M', time.localtime(time_modify))
                table_line = [colored(hour_min_str, attrs=['bold']), task_name,
                              max_miou_str, max_mf1_str, metrics['epoch'][-1], time_modify_str]

                if int(metrics['epoch'][-1]) == self.max_epoch:
                    if (time.time() - time_modify) > 1440 * 60:
                        color = 'blue'
                    else:
                        color = 'green'
                        if (time.time() - time_modify) < 5 * 60:
                            self.send_tele(table_line)
                else:
                    if (time.time() - time_modify) > self.wait_min * 60:
                        color = 'red'
                    else:
                        color = 'white'
                new_table_line = []
                for i, word in enumerate(table_line):
                    if i > 0:
                        word = colored(word, color)
                    new_table_line.append(word)
                table_line = new_table_line

                data_raw[month_day].append(table_line)

            except:
                continue

        for k, v in data_raw.items():
            for i, line in enumerate(v):
                if i == 0:
                    line.insert(0, k)
                else:
                    line.insert(0, '')
                table.append(line)

        headers = ['day', 'time', 'task', 'miou', 'mf1', 'cur', 'update']
        os.system('clear')
        print(tabulate(table, headers=headers, tablefmt='presto',
                       stralign='left', numalign='right'))


if __name__ == '__main__':
    # path = '../lane_detection/work_dirs'
    # follow_links = False
    path = '/private/personal/liuzili/workspace/lane_detection/work_dirs/logs'
    follow_links = True

    recent_days = 60
    max_epoch = 11
    wait_min = 180

    monitor = Monitor(path, recent_days, follow_links, max_epoch, wait_min)

    while True:
        monitor.stat_logs()
        time.sleep(30)
