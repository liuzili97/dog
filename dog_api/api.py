from .dist_util import master_only
from .summary import (
    set_batch_per_gpu, set_writer, set_max_epochs, set_max_inner_iters, set_epoch,
    before_epoch, after_train_epoch, after_val_epoch, after_train_iter, set_inner_iter,
    add_coco_eval_summary, every_n_inner_iters
)
from .task import add_device_info, update_task_info
from .env import dog_launched
from .util import occumpy_mem


@dog_launched
def dog_init(output_dir, batch_per_gpu, max_epochs, max_inner_iters):
    occumpy_mem()
    add_device_info()

    set_batch_per_gpu(batch_per_gpu)
    set_max_epochs(max_epochs)
    set_max_inner_iters(max_inner_iters)

    # master-only
    set_writer(f"{output_dir}")
    update_task_info(dict(max_epochs=max_epochs, max_inner_iters=max_inner_iters))


@dog_launched
@master_only
def dog_before_epoch(epoch):
    set_epoch(epoch)

    update_task_info(dict(epoch=epoch))
    before_epoch()


@dog_launched
@master_only
def dog_after_train_epoch():
    after_train_epoch()


@dog_launched
@master_only
def dog_after_val_epoch(coco_eval_list=None):
    if coco_eval_list:
        add_coco_eval_summary(coco_eval_list)
    after_val_epoch()


@dog_launched
@master_only
def dog_before_train_iter(inner_iter):
    set_inner_iter(inner_iter)

    if every_n_inner_iters(20):
        update_task_info(dict(inner_iter=inner_iter))


@dog_launched
@master_only
def dog_after_train_iter():
    after_train_iter()
