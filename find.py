import os
import sys
from termcolor import colored
from tabulate import tabulate

from loader import TargetLoader
from dog_api import dog


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
    summaries = dog.summary_dog.load_summary_files()
    for k, v in summaries.items():
        if keyword in v['name'] and v['eta'] == 'Done':
            os.system(f"rm {k}")


def main():
    args = sys.argv[1:]
    assert args == 1
    master_name = args[:1]
    target_loader = TargetLoader(master_name)

    args = sys.argv[1:]
    assert len(args) == 1, args
    keyword = args[0]

    target_dirs = target_loader.get_target_dirs()

    show_find_res(target_dirs, keyword)
    remove_dog_file(keyword)


if __name__ == '__main__':
    main()
