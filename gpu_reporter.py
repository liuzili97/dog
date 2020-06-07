import time
import json
import os
import socket
import paramiko
import threading
from termcolor import colored
from tabulate import tabulate
from collections import OrderedDict
from utils import Loader

home = os.environ['HOME']
disp_thre = dict(used=0.2, power=60)
all_cmds = dict(
    gpus='nvidia-smi --query-gpu=memory.used,memory.total,power.draw --format=csv,noheader',
    # tasks='cat {}/.watch_dog/{{}}_*'.format(known_nodes['home'])
)
mark = '|'

_gpus_results = {}
_tasks_results = {}
_sshs = {}


def ssh(name, connect_args, cmds):
    try:
        if name not in _sshs:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(**connect_args)
            _sshs[name] = ssh
        else:
            ssh = _sshs[name]
        _gpus_results[name] = ssh.exec_command(cmds['gpus'])[1].readlines()
        _tasks_results[name] = ''
        # _, stdout, _ = ssh.exec_command(cmds['tasks'].format(node_name))
        # _tasks_results[node_name] = stdout.readlines()
    except:
        _gpus_results[name] = ''
        _tasks_results[name] = ''
    # ssh.close()


def get_prog_disp_info(memory_used, power):
    much_used = memory_used > disp_thre['used']
    much_power = power > disp_thre['power']
    is_free = 0
    if not much_used and not much_power:
        prog_col = 'green'
        is_free = 1
    elif much_used:
        prog_col = 'red'
    elif much_power:
        prog_col = 'yellow'
    return prog_col, is_free


def get_name_disp_info(gpu_num, free_gpu_num):
    suggest_level = 0
    if 0 < gpu_num == free_gpu_num:
        name_col = dict(color='green', attrs=['bold'])
        suggest_level = 3
    elif 0 < free_gpu_num < gpu_num:
        if free_gpu_num / gpu_num >= 0.5:
            name_col = dict(color='cyan', attrs=[])
            suggest_level = 2
        else:
            name_col = dict(color='yellow', attrs=[])
            suggest_level = 1
    elif gpu_num == 0:
        name_col = dict(color='white', attrs=['dark'])
    else:
        name_col = dict(color='white', attrs=[])
    return name_col, suggest_level


def single_gpu(gpu_info):
    # get ['449', '32510', '55.05'] from '449 MiB, 32510 MiB, 55.05 W'
    gpu_info = [item.split()[0] for item in gpu_info.split(',')]
    for i, item in enumerate(gpu_info):
        # return [Not Supported] for some gpus
        gpu_info[i] = float(item) if 'No' not in item else 0.

    memory_used = gpu_info[0] / gpu_info[1]
    power = gpu_info[2]
    prog_col, is_free = get_prog_disp_info(memory_used, power)

    percent = int(memory_used * 10)
    gpu_str = colored(mark, prog_col, attrs=['bold']) + \
              colored(mark * percent + ' ' * (9 - percent), prog_col) + \
              colored('|', attrs=['dark'])
    return gpu_str, is_free



def single_node(node_name, res):
    # tasks_res = _tasks_results[node_name]
    gpus_info = []
    name_col = {}
    suggest_level = 0

    if isinstance(res, list) and res:
        gpu_num, free_gpu_num = len(res), 0

        for gpu_id, gpu_info in enumerate(res):
            gpu_str, is_free = single_gpu(gpu_info)
            gpus_info.append(gpu_str)
            free_gpu_num += is_free

        name_col, suggest_level = get_name_disp_info(gpu_num, free_gpu_num)

        # if _tasks_results[node_name]:
        #     col['attrs'].append('underline')

    return [colored(node_name, **name_col)] + gpus_info, suggest_level


def init_headers(node_names):
    gpu_headers = ['node'] + ['GPU{}'.format(i) for i in range(8)]
    tasks_headers = ['node', 'tasks_names']
    return gpu_headers, tasks_headers


if __name__ == '__main__':
    hostname = socket.gethostname()
    loader = Loader(hostname)
    connect_dict = loader.get_connect_dict()

    while True:
        threads = []
        for name, connect_args in connect_dict.items():
            threads.append(
                threading.Thread(target=ssh, args=(name, connect_args, all_cmds)))
        for thr in threads:
            thr.start()
        for thr in threads:
            thr.join()

        data = []
        summary = dict(node_l3=[], node_l2=[], node_l1=[], node_l0=[])
        for name in loader.get_all_nodes():
            info, suggest_level = single_node(name, _gpus_results[name])
            data.append(info)
            summary['node_l{}'.format(suggest_level)].append(name)

        headers = init_headers(loader.get_all_nodes())
        table = tabulate(data, headers=headers[0], tablefmt='rst')
        summary_msg = colored(
            "All free ({}):\t{}".format(len(summary['node_l3']), '  '.join(summary['node_l3'])),
            'green', attrs=['bold']) + colored(
            "\nMost free ({}):\t{}".format(len(summary['node_l2']),
                                           '  '.join(summary['node_l2'])),
            'cyan') + colored(
            "\nSome free ({}):\t{}\n".format(len(summary['node_l1']),
                                             '  '.join(summary['node_l1'])),
            'yellow')

        # tasks_data = []
        # for name in known_nodes['name']:
        #     tasks_res = _tasks_results[name]
        #     if tasks_res:
        #         new_tasks_res = []
        #         for t in tasks_res:
        #             t = t.strip().split()
        #             eta = '' if len(t) < 4 else t[3]
        #             t = t[:2]
        #
        #             tmp = colored(t[-1], 'green')
        #             for key in ['moco', 'coco', 'imagenet', 'r18', 'r34', 'r50', 'd53', 'ttf']:
        #                 if key in t[-1]:
        #                     tmp = colored(t[-1], 'blue')
        #             t[-1] = tmp
        #             t.append(colored(eta, 'cyan'))
        #             t = ' '.join(t)
        #             new_tasks_res.append(t)
        #         tasks_res = '\n'.join(new_tasks_res)
        #         tasks_data.append([name, tasks_res])
        # tasks_table = tabulate(tasks_data, tablefmt='grid', showindex=True)

        os.system('clear')
        print(table)
        print(summary_msg)
        # print(tasks_table)
        time.sleep(10)
