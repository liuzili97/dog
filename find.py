import os
import sys
import socket
from termcolor import colored
from tabulate import tabulate

from loader import CFGLoader
from dog_utils import load_dog_file

dog_dir = '{}/.dog'.format(os.environ['HOME'])


def show_find_res(target_dirs, keyword):
    find_res, find_res_show = [], []
    for target_dir in target_dirs:
        root_dir = os.path.join(target_dir, 'work_dirs')
        for root, dirs, files in os.walk(root_dir):
            if dirs:
                for dir_name in dirs:
                    if keyword in dir_name:
                        find_res.append([os.path.join(root, dir_name)])

                        short_root = root.replace(os.environ['HOME'], '~')
                        color_root = short_root.replace(keyword, colored(keyword, 'green'))
                        color_dir_name = dir_name.replace(keyword, colored(keyword, 'cyan'))
                        find_res_show.append([os.path.join(color_root, color_dir_name)])

    print(tabulate(find_res_show, tablefmt='grid', showindex=True))


def remove_dog_file(keyword):
    for root, _, files in os.walk(dog_dir):
        for file in files:
            task_dict = load_dog_file(file)
            if keyword in task_dict['name'] and task_dict['eta'] == 'Done':
                os.system('rm {}'.format(os.path.join(root, file)))


def main():
    hostname = socket.gethostname()
    loader = CFGLoader(hostname)

    args = sys.argv[1:]
    assert len(args) == 1, args
    keyword = args[0]

    target_dirs = loader.get_target_dirs()

    show_find_res(target_dirs, keyword)
    remove_dog_file(keyword)


if __name__ == '__main__':
    main()
