import os
import json
import functools
import time
import numpy as np
from datetime import datetime
from collections import OrderedDict
from termcolor import colored

from torch.utils.tensorboard import SummaryWriter

from .dist_utils import master_only
from .utils import occupy_mem, is_use_slurm
from loader import SettingLoader


def dog_launched(old_func):
    @functools.wraps(old_func)
    def new_func(*args, **kwargs):
        if os.environ.get('DOG_LAUNCHED', 'False') != 'True':
            print(colored("NOT LAUNCHED", 'red'))
            return

        output = old_func(*args, **kwargs)
        return output

    return new_func


class BaseDog:

    def __init__(self):
        setting_loader = SettingLoader()
        self.write_interval = setting_loader.get_write_interval()
        self.summary_dir = os.path.join(setting_loader.get_basedir(),
                                        setting_loader.get_dirname())

        if not os.path.isdir(self.summary_dir):
            os.makedirs(self.summary_dir)

    def init(self):
        pass

    def before_run(self):
        pass

    def before_train_epoch(self, epoch):
        pass

    def after_train_epoch(self):
        pass

    def before_val_epoch(self):
        pass

    def after_val_epoch(self):
        pass

    def before_train_iter(self, inner_iter):
        pass

    def after_train_iter(self):
        pass

    def every_n_inner_iters(self, inner_iter):
        return (inner_iter + 1) % self.write_interval == 0


class SummaryDog(BaseDog):
    # run on node. init when a task is launched by the dog
    def __init__(self):
        super(SummaryDog, self).__init__()
        self.summary_file = None
        self.summary_dict = dict()

    def register_task(self, task_name, gpu_info):
        # called in main
        gpu_num = len(gpu_info) if isinstance(gpu_info, list) else int(gpu_info)
        summary_key = datetime.now().strftime('%m.%d-%H:%M:%S')
        self.summary_file = os.path.join(self.summary_dir, summary_key)
        self.summary_dict = dict(key=summary_key, name=task_name,
                                 eta='Starting', gpu_num=gpu_num)

        # since register_task is not executed in workers
        # we need to store the vars and restore it
        os.environ['DOG_SUMMARY_FILE'] = self.summary_file
        os.environ['DOG_LAUNCHED'] = 'True'
        self.write_out()

    def finish_task(self):
        # called in main
        self.summary_dict = self.load_summary_file()
        self.add_summary(eta='Done')
        self.write_out()
        del os.environ['DOG_LAUNCHED']
        self.rm_cache()

    @dog_launched
    def init(self):
        # restore vars for workers since they will not execute register_task
        self.summary_file = os.environ['DOG_SUMMARY_FILE']
        occupy_mem()

    @dog_launched
    @master_only
    def before_run(self, max_epochs, max_inner_iters):
        self.summary_dict = self.load_summary_file()
        if is_use_slurm():
            self.add_summary(name=f"{self.summary_dict['name']} ({os.environ['SLURM_JOB_ID']})")
        self.add_summary(max_epochs=max_epochs, max_inner_iters=max_inner_iters)
        self.write_out()

    @dog_launched
    @master_only
    def before_train_epoch(self, epoch):
        self.add_summary(epoch=epoch)
        self.write_out()

    @dog_launched
    @master_only
    def after_val_epoch(self, coco_eval_bbox=None):
        if coco_eval_bbox:
            self.add_coco_eval_summary(coco_eval_bbox)
        self.write_out()

    @dog_launched
    @master_only
    def before_train_iter(self, inner_iter):
        self.add_summary(inner_iter=inner_iter)

    @dog_launched
    @master_only
    def after_train_iter(self):
        if self.every_n_inner_iters(self.summary_dict['inner_iter']):
            self.write_out()

    def get_task_eta(self, task_name):
        # called by dog
        summaries = self.load_summary_files()
        for v in summaries.values():
            if v['name'] == task_name:
                return v['eta']
        return 'Done'

    def rm_cache(self):
        # called by dog
        print(self.summary_dict)

        key = self.summary_dict['key']
        start_stamp = time.mktime(time.strptime(key, '%m.%d-%H:%M:%S'))
        now_stamp = time.mktime(time.strptime(
            datetime.now().strftime('%m.%d-%H:%M:%S'), '%m.%d-%H:%M:%S'))
        if (now_stamp - start_stamp) / 60 < 5:
            os.system(f"rm {self.summary_file}")

    @dog_launched
    @master_only
    def add_summary(self, **kwargs):
        for k, v in kwargs.items():
            self.summary_dict[k] = v

    @dog_launched
    @master_only
    def add_coco_eval_summary(self, eval_bbox):
        assert len(eval_bbox) == 12
        self.add_summary(
            AP=f"{eval_bbox[0]:.04f}", AP_s=f"{eval_bbox[3]:.04f}",
            AP_m=f"{eval_bbox[4]:.04f}", AP_l=f"{eval_bbox[5]:.04f}")

    @dog_launched
    @master_only
    def write_out(self):
        self.summary_dict['last_update'] = datetime.now().strftime('%d-%H%M')
        with open(self.summary_file, 'w') as f:
            f.write(json.dumps(self.summary_dict, indent=4))

    @dog_launched
    @master_only
    def load_summary_file(self):
        return json.loads(os.popen(f"cat {self.summary_file}").read())

    @master_only
    def load_summary_files(self):
        summaries = {}
        for root, _, files in os.walk(self.summary_dir):
            for file in files:
                summary_file = os.path.join(root, file)
                summaries[summary_file] = json.loads(os.popen(f"cat {summary_file}").read())
        return summaries


class BoardDog(BaseDog):

    def __init__(self):
        super(BoardDog, self).__init__()
        self.writer = None
        self.smooth_board = LogBuffer()

        self.epoch = 0
        self.max_epochs = 0
        self.inner_iter = 0
        self.max_inner_iters = 0
        self.batch_per_gpu = 1
        self.gpu_num = 0

    @dog_launched
    @master_only
    def before_run(self, max_epochs, max_inner_iters, output_dir, batch_per_gpu, gpu_num):
        self.writer = SummaryWriter(output_dir)
        self.max_epochs = max_epochs
        self.max_inner_iters = max_inner_iters
        self.batch_per_gpu = batch_per_gpu
        self.gpu_num = gpu_num

    @dog_launched
    @master_only
    def before_train_epoch(self, epoch):
        self.epoch = epoch
        self.smooth_board.clear()

    @dog_launched
    @master_only
    def after_train_epoch(self):
        self.smooth_board.average(self.write_interval)
        if self.smooth_board.ready:
            self.write_out()
        self.smooth_board.clear_output()

    @dog_launched
    @master_only
    def after_val_epoch(self, coco_eval_bbox=None):
        if coco_eval_bbox:
            self.add_coco_eval_summary(coco_eval_bbox)
        self.write_out()
        self.smooth_board.clear_output()

    @dog_launched
    @master_only
    def before_train_iter(self, inner_iter):
        self.inner_iter = inner_iter

    @dog_launched
    @master_only
    def after_train_iter(self):
        if self.every_n_inner_iters(self.inner_iter):
            self.smooth_board.average(self.write_interval)

        if self.smooth_board.ready:
            self.write_out()
            self.smooth_board.clear_output()

    @dog_launched
    @master_only
    def add_summary(self, prefix, **kwargs):
        board_dict = dict()
        for k, v in kwargs.items():
            board_dict[f'{prefix}/{k}'] = v
        self.smooth_board.update(board_dict)

    @dog_launched
    @master_only
    def add_coco_eval_summary(self, eval_bbox):
        assert len(eval_bbox) == 12
        AP = dict(AP=eval_bbox[0], AP_5=eval_bbox[1], AP_75=eval_bbox[2], AP_s=eval_bbox[3],
                  AP_m=eval_bbox[4], AP_l=eval_bbox[5])
        AR = dict(AR_1=eval_bbox[6], AR_10=eval_bbox[7], AR_100=eval_bbox[8], AR_s=eval_bbox[9],
                  AR_m=eval_bbox[10], AR_l=eval_bbox[11])
        for k, v in AP.items():
            self.smooth_board.output[f'AP/{k}'] = v
        for k, v in AR.items():
            self.smooth_board.output[f'AR/{k}'] = v
        self.smooth_board.ready = True

    @dog_launched
    @master_only
    def set_epoch(self, epoch):
        # start from 1
        self.epoch = epoch

    @dog_launched
    @master_only
    def set_inner_iter(self, inner_iter):
        self.inner_iter = inner_iter

    @dog_launched
    @master_only
    def set_gpu_num(self, gpu_num):
        self.gpu_num = gpu_num

    @dog_launched
    @master_only
    def get_iter_left(self):
        return (self.max_epochs - self.epoch) * self.max_inner_iters + \
               (self.max_inner_iters - self.inner_iter)

    @dog_launched
    @master_only
    def get_global_step(self):
        return ((self.epoch - 1) * self.max_inner_iters + self.inner_iter) * \
               self.batch_per_gpu * self.gpu_num

    @dog_launched
    @master_only
    def write_out(self):
        for k, v in self.smooth_board.output.items():
            self.writer.add_scalar(k, v, self.get_global_step())


class FullDog(BaseDog):

    def __init__(self):
        super(BaseDog, self).__init__()
        self.summary_dog = SummaryDog()
        self.board_dog = BoardDog()

    @dog_launched
    def init(self):
        self.summary_dog.init()

    @dog_launched
    @master_only
    def before_run(self, max_epochs, max_inner_iters, output_dir, batch_per_gpu):
        self.summary_dog.before_run(max_epochs, max_inner_iters)
        gpu_num = self.summary_dog.summary_dict['gpu_num']
        self.board_dog.before_run(max_epochs, max_inner_iters, output_dir, batch_per_gpu,
                                  gpu_num)

    @dog_launched
    @master_only
    def before_train_epoch(self, epoch):
        self.summary_dog.before_train_epoch(epoch)
        self.board_dog.before_train_epoch(epoch)

    @dog_launched
    @master_only
    def after_train_epoch(self):
        self.board_dog.after_train_epoch()

    @dog_launched
    @master_only
    def after_val_epoch(self, coco_eval_bbox=None):
        self.summary_dog.after_val_epoch(coco_eval_bbox=coco_eval_bbox)
        self.board_dog.after_val_epoch(coco_eval_bbox=coco_eval_bbox)

    @dog_launched
    @master_only
    def before_train_iter(self, inner_iter):
        self.summary_dog.before_train_iter(inner_iter)
        self.board_dog.before_train_iter(inner_iter)

    @dog_launched
    @master_only
    def after_train_iter(self):
        self.summary_dog.after_train_iter()
        self.board_dog.after_train_iter()


class LogBuffer(object):

    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False
        self.max_record = 1000

    def clear(self):
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self):
        self.output.clear()
        self.ready = False

    def update(self, vars, count=1):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def update_output(self, vars):
        assert isinstance(vars, dict)
        for key, var in vars.items():
            self.output[key] = var

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg

            if len(self.val_history[key]) > self.max_record:
                self.val_history[key] = self.val_history[key][-self.max_record:]
                self.n_history[key] = self.n_history[key][-self.max_record:]
        self.ready = True
