import time
import sys
import torch

from dog_api import check_mem


def occumpy_mem(device_id):
    total, used = check_mem(device_id)
    max_mem = int(total * 0.92)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem, device=int(device_id))
    del x


if __name__ == '__main__':
    assert len(sys.argv) == 2
    device_ids = sys.argv[1].split(',')
    for did in device_ids:
        occumpy_mem(did)
    time.sleep(600)
