import sys
import os

from loader import MasterLoader

if __name__ == '__main__':
    args = sys.argv[1:]
    node_name = args[0]
    cpu_num = args[1]

    loader = MasterLoader('tc6000')
    partition = loader.get_slurm_env(node_name)['PARTITION']
    os.environ['SSH_PART'] = partition
    os.system(f"salloc -p {partition} -w {node_name} --cpus-per-task={cpu_num}")
