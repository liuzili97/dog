from termcolor import colored

from torch.utils.tensorboard import SummaryWriter

from .dist_util import master_only
from .log_buffer import LogBuffer
from .task import update_task_info


_summary_dict = dict()
_writer = None
_log_buffer = LogBuffer()
_global_step = 0
_local_step = 0
_epoch = 0
_max_epochs = 0
_inner_iter = 0
_max_inner_iters = 1
_batch_per_gpu = 1
_gpu_num = 1
_default_summary_freq = 1000


@master_only
def add_summary(prefix, average_factor=1, **kwargs):
    global _summary_dict
    for var, value in kwargs.items():
        if prefix:
            _summary_dict['{}/{}'.format(prefix, var)] = value / average_factor
        else:
            _summary_dict['{}'.format(var)] = value / average_factor


@master_only
def add_coco_eval_summary(eval_bbox):
    global _log_buffer
    assert len(eval_bbox) == 12
    AP = dict(AP=eval_bbox[0], AP_5=eval_bbox[1], AP_75=eval_bbox[2], AP_s=eval_bbox[3],
              AP_m=eval_bbox[4], AP_l=eval_bbox[5])
    AR = dict(AR_1=eval_bbox[6], AR_10=eval_bbox[7], AR_100=eval_bbox[8], AR_s=eval_bbox[9],
              AR_m=eval_bbox[10], AR_l=eval_bbox[11])
    update_task_info(dict(
        AP=f"{eval_bbox[0]:.04f}", AP_s=f"{eval_bbox[3]:.04f}",
        AP_m=f"{eval_bbox[4]:.04f}", AP_l=f"{eval_bbox[5]:.04f}"))
    for k, v in AP.items():
        _log_buffer.output[f'AP/{k}'] = v
    for k, v in AR.items():
        _log_buffer.output[f'AR/{k}'] = v
    _log_buffer.ready = True


def get_summary():
    return _summary_dict


@master_only
def set_writer(dir_name='./tmp'):
    global _writer
    _writer = SummaryWriter(dir_name)
    return _writer


def get_writer():
    global _writer
    assert _writer, "writer is not init."
    return _writer


def reset_global_step(global_step=0):
    global _global_step
    _global_step = global_step
    print(colored(f"Global step has been set to {global_step}", 'red'))


def update_global_step(delta_step=1):
    global _global_step
    global _local_step
    _global_step += delta_step
    _local_step += 1


def set_epoch(epoch):
    # start from 1
    global _epoch
    _epoch = epoch


def set_max_epochs(max_epochs):
    global _max_epochs
    _max_epochs = max_epochs


def set_inner_iter(inner_iter):
    global _inner_iter
    _inner_iter = inner_iter


def set_max_inner_iters(max_inner_iters):
    global _max_inner_iters
    _max_inner_iters= max_inner_iters


def set_gpu_num(gpu_num):
    global _gpu_num
    _gpu_num = gpu_num


def set_batch_per_gpu(batch_per_gpu):
    global _batch_per_gpu
    _batch_per_gpu = batch_per_gpu


def get_global_step():
    return _global_step


def get_local_step():
    return _local_step


def get_epoch():
    return _epoch


def get_max_epochs():
    return _max_epochs


def get_inner_iter():
    return _inner_iter


def get_max_inner_iters():
    return _max_inner_iters


def get_iter_left():
    return (_max_epochs - _epoch) * _max_inner_iters + (_max_inner_iters - _inner_iter)


def every_n_inner_iters(n):
    return (_inner_iter + 1) % n == 0 if n > 0 else False


@master_only
def every_n_local_step(n=_default_summary_freq):
    if _local_step > 0 and _local_step % n == 0:
        return True
    return False


def log(log_buffer):
    for var in log_buffer.output:
        tag = '{}'.format(var)
        _writer.add_scalar(tag, log_buffer.output[var], get_global_step())


def update_log_buffer(var, smooth=True):
    """

    :param var: dict, watch list var
    :param smooth: bool, if smooth, will use log buffer's update, else
    add var in output directly
    :return: None
    """
    global _log_buffer
    if smooth:
        _log_buffer.update(var)
    else:
        _log_buffer.update_output(var)


@master_only
def before_epoch():
    _log_buffer.clear()


@master_only
def after_train_iter(interval=20):
    global _global_step, _local_step
    update_log_buffer(get_summary())
    _global_step = ((_epoch - 1) * _max_inner_iters + _inner_iter) * _batch_per_gpu * _gpu_num
    _local_step += 1

    if every_n_inner_iters(interval):
        _log_buffer.average(interval)

    if _log_buffer.ready:
        log(_log_buffer)
        _log_buffer.clear_output()


@master_only
def after_train_epoch():
    _log_buffer.average()
    if _log_buffer.ready:
        log(_log_buffer)
    _log_buffer.clear_output()


@master_only
def after_val_epoch():
    update_log_buffer(get_summary())

    log(_log_buffer)
    _log_buffer.clear_output()
