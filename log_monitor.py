import os
import time
import numpy as np

from tabulate import tabulate
from collections import defaultdict
from termcolor import colored


def search_log(base_dir, recent_days=2):
    def taketime(elem):
        return elem[0]

    log_files = []
    for root, _, files in os.walk(base_dir, followlinks=True):
        for file in files:
            if file.endswith('.log'):
                time_pass = time.time() - time.mktime(
                    time.strptime(file.split('.')[0], "%Y%m%d_%H%M%S"))
                if time_pass < 3600 * 24 * recent_days:
                    log_files.append((time_pass, os.path.join(root, file)))

    log_files.sort(key=taketime, reverse=True)
    log_files = [log_file[1] for log_file in log_files]
    return log_files


def stat_logs(log_files):
    table = []
    data_raw = defaultdict(list)
    for log_file in log_files:
        try:
            lines = [line for line in open(log_file, 'r').readlines() if 'Overall' in line]
        except:
            continue
        metrics = defaultdict(list)
        if len(lines) == 0:
            continue
        for line in lines:
            splitted_line = line.split()
            metrics['mAP_25'].append(splitted_line[3])
            metrics['mAP_50'].append(splitted_line[7])
        metrics['mAP_25'] = np.array(metrics['mAP_25']).astype(np.float32)
        metrics['mAP_50'] = np.array(metrics['mAP_50']).astype(np.float32)
        max_25 = metrics['mAP_25'].max() * 100
        max_50 = metrics['mAP_50'].max() * 100
        max_25_str = f"{max_25:5.2f} ({metrics['mAP_25'].argmax():>2d})"
        max_50_str = f"{max_50:5.2f} ({metrics['mAP_50'].argmax():>2d})"
        task_name = f'{log_file.split("/")[-3]}'
        time_str = time.strftime('%m%d-%H%M',
                                 time.localtime(time.mktime(time.strptime(
                                     os.path.basename(log_file)[:-4], "%Y%m%d_%H%M%S"))))
        month_day = time_str.split('-')[0]
        hour_min_str = time_str.split('-')[1]

        time_modify = os.path.getmtime(log_file)
        time_modify_str = time.strftime('%d-%H%M', time.localtime(time_modify))
        table_line = [colored(hour_min_str, attrs=['bold']), task_name, f"{max_25_str}",
                      f"{max_50_str}", metrics['mAP_50'].shape[0], time_modify_str]
        if metrics['mAP_50'].shape[0] < 36:
            color = 'blue'
            if (time.time() - time_modify) > 2 * 60:
                color = 'red'
            new_table_line = []
            for i, word in enumerate(table_line):
                if i > 0:
                    word = colored(word, color)
                new_table_line.append(word)
            table_line = new_table_line

        data_raw[month_day].append(table_line)

    for k, v in data_raw.items():
        for i, line in enumerate(v):
            if i == 0:
                line.insert(0, k)
            else:
                line.insert(0, '')
            table.append(line)

    headers = ['day', 'time', 'task', 'AP25', 'AP50', 'cur', 'update']
    os.system('clear')
    print(tabulate(table, headers=headers, tablefmt='presto',
                   stralign='left', numalign='right'))



if __name__ == '__main__':
    while True:
        log_files = search_log('../work_dirs', recent_days=60)
        stat_logs(log_files)
        time.sleep(30)
