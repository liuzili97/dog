import os

import numpy as np


def get_gpu_info(keyword):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep {} >tmp_gpu'.format(keyword.title()))
    return np.array([int(x.split()[2]) for x in open('tmp_gpu', 'r').readlines()])


def parse(line, qargs):
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
    power_manage_enable = lambda v: (not 'Not Support' in v)
    to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').replace('W', ''))
    process = lambda k, v: ((int(to_numberic(v)) if power_manage_enable(v) else 1)
                            if k in numberic_args else v.strip())
    return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}


def query_gpu():
    qargs = ['index', 'memory.free', 'memory.total', 'power.draw']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return [parse(line, qargs) for line in results]


def get_free_gpu(thre=0.9):
    gpus_info = query_gpu()
    gpu_num = len(gpus_info)
    gpu_free = [False for _ in range(gpu_num)]
    for gpu_info in gpus_info:
        gpu_id = int(gpu_info['index'])
        gpu_free[gpu_id] = float(gpu_info['memory.free']) / float(gpu_info['memory.total']) > thre
    gpu_free_id = [str(idx) for idx in range(gpu_num) if gpu_free[idx]]
    return gpu_free_id, gpu_num
