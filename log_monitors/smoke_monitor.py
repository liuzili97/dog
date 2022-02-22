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


def stat_logs(log_files, max_epoch, wait_min):
    table = []
    data_raw = defaultdict(list)
    for log_file in log_files:
        try:
            lines = open(log_file, 'r').readlines()

            metrics = defaultdict(list)
            if len(lines) == 0:
                continue
            for i, line in enumerate(lines):
                if 'bev  AP:' in line and 'Car AP@0.70, 0.70, 0.70' in lines[i - 2]:
                    splitted_line = line.split(':')[1].strip().split(',')
                    metrics['bev_easy'].append(splitted_line[0])
                    metrics['bev_mid'].append(splitted_line[1])
                    metrics['bev_hard'].append(splitted_line[2])

                    splitted_line = lines[i + 1].split(':')[1].strip().split(',')
                    metrics['3d_easy'].append(splitted_line[0])
                    metrics['3d_mid'].append(splitted_line[1])
                    metrics['3d_hard'].append(splitted_line[2])

            metrics['bev_easy'] = np.array(metrics['bev_easy']).astype(np.float32)
            metrics['bev_mid'] = np.array(metrics['bev_mid']).astype(np.float32)
            metrics['bev_hard'] = np.array(metrics['bev_hard']).astype(np.float32)
            metrics['3d_easy'] = np.array(metrics['3d_easy']).astype(np.float32)
            metrics['3d_mid'] = np.array(metrics['3d_mid']).astype(np.float32)
            metrics['3d_hard'] = np.array(metrics['3d_hard']).astype(np.float32)

            max_bev_easy = metrics['bev_easy'].max()
            max_bev_mid = metrics['bev_mid'].max()
            max_bev_hard = metrics['bev_hard'].max()
            max_3d_easy = metrics['3d_easy'].max()
            max_3d_mid = metrics['3d_mid'].max()
            max_3d_hard = metrics['3d_hard'].max()

            max_bev_easy_str = f"{max_bev_easy:5.2f} ({metrics['bev_easy'].argmax():>2d})"
            max_bev_mid_str = f"{max_bev_mid:5.2f} ({metrics['bev_mid'].argmax():>2d})"
            max_bev_hard_str = f"{max_bev_hard:5.2f} ({metrics['bev_hard'].argmax():>2d})"
            max_3d_easy_str = f"{max_3d_easy:5.2f} ({metrics['3d_easy'].argmax():>2d})"
            max_3d_mid_str = f"{max_3d_mid:5.2f} ({metrics['3d_mid'].argmax():>2d})"
            max_3d_hard_str = f"{max_3d_hard:5.2f} ({metrics['3d_hard'].argmax():>2d})"

            task_name = f'{log_file.split("/")[-3]}'
            time_str = time.strftime('%m%d-%H%M',
                                     time.localtime(time.mktime(time.strptime(
                                         os.path.basename(log_file)[:-4], "%Y%m%d_%H%M%S"))))
            month_day = time_str.split('-')[0]
            hour_min_str = time_str.split('-')[1]

            time_modify = os.path.getmtime(log_file)
            time_modify_str = time.strftime('%d-%H%M', time.localtime(time_modify))
            table_line = [colored(hour_min_str, attrs=['bold']), task_name, max_bev_easy_str,
                          max_bev_mid_str, max_bev_hard_str, max_3d_easy_str, max_3d_mid_str,
                          max_3d_hard_str, metrics['3d_hard'].shape[0], time_modify_str]
            if metrics['3d_hard'].shape[0] < max_epoch:
                color = 'blue'
                if (time.time() - time_modify) > wait_min * 60:
                    color = 'red'
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

    headers = ['day', 'time', 'task', 'BEV_easy', 'BEV_mid', 'BEV_hard', '3D_easy',
               '3D_mid', '3D_hard', 'cur', 'update']
    os.system('clear')
    print(tabulate(table, headers=headers, tablefmt='presto',
                   stralign='left', numalign='right'))



if __name__ == '__main__':
    max_epoch = 100
    wait_min = 20
    while True:
        log_files = search_log('../MonoFlex/work_dirs', recent_days=60)
        stat_logs(log_files, max_epoch, wait_min)
        time.sleep(30)
