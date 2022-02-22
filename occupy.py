import time
import sys
import torch

from dog_api import check_mem


def occupy_mem(device_id):
    _, free = check_mem(device_id)
    block_mem = int(free * 0.9)
    x = torch.cuda.FloatTensor(256, 1024, block_mem, device=int(device_id))
    del x


if __name__ == '__main__':
    assert len(sys.argv) == 2
    device_ids = sys.argv[1].split(',')
    while True:
        for did in device_ids:
            occupy_mem(did)
        time.sleep(60)
        torch.cuda.empty_cache()
