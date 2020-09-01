import os
import json
import mmcv
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F
import pickle
import time
import socket
from multiprocessing import Pool
from PIL import Image, ImageStat
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val
from mmcv.visualization.image import imshow
from pycocotools.coco import COCO


def decode_bbox(bbox):
    x1, y1, w, h = bbox
    return x1, y1, x1 + w, y1 + h


def encode_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def random_select(l, n):
    rand_ids = list(range(len(l)))
    random.shuffle(rand_ids)
    return [l[i] for i in rand_ids[:n]]


class BGMemoryBuilder:

    def __init__(self, json_path, dataset_path, gen_path,
                 use_seg=True, paste_min_size=10, fail_times=50, max_bg=10):
        self.dataset_path = dataset_path
        self.gen_path = gen_path
        self.use_seg = use_seg
        self.paste_min_size = paste_min_size
        self.fail_times = fail_times
        self.max_bg = max_bg

        if not os.path.isdir(self.gen_path):
            os.makedirs(self.gen_path)

        self.json_file = json.load(open(json_path, 'r'))
        self.coco = COCO(json_path)
        self.images = self.json_file['images']
        self.raw_annotations = self.json_file['annotations']
        self.annotations = [ann for ann in self.raw_annotations if ann['iscrowd'] == 0]
        self.crowd_annotations = [ann for ann in self.raw_annotations if ann['iscrowd'] == 1]

        self.image_id_to_ann_idxs = defaultdict(list)

        for idx, ann in enumerate(self.annotations):
            self.image_id_to_ann_idxs[ann['image_id']].append(idx)

    def get_target_mask(self, img):
        mask = np.zeros((img['height'], img['width']), dtype=np.bool)
        for ann_idx in self.image_id_to_ann_idxs[img['id']]:
            ann = self.annotations[ann_idx]
            if self.use_seg:
                fg_mask = self.coco.annToMask(ann).astype(np.bool)
                mask[fg_mask == 1] = 1
            else:
                x1, y1, x2, y2 = decode_bbox(ann['bbox'])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                mask[y1:y2 + 1, x1:x2 + 1] = 1
        return mask

    def gen_bg(self):
        img_shapes = dict()
        for img in tqdm(self.images):
            target_mask = self.get_target_mask(img)
            if target_mask.sum() / (img['height'] * img['width']) > 0.8:
                continue
            i = 0
            j = 0

            target_h, target_w = target_mask.shape
            bboxes = []
            while i < self.fail_times and j < self.max_bg:
                i += 1
                x1 = random.randint(0, target_w - 30)
                y1 = random.randint(0, target_h - 30)
                x2 = random.randint(x1 + 10, target_w)
                y2 = random.randint(y1 + 10, target_h)
                if target_mask[y1:y2 + 1, x1:x2 + 1].any():
                    continue
                bboxes.append((x1, y1, x2, y2))
                j += 1

            if len(bboxes) > 0:
                image = mmcv.imread(os.path.join(self.dataset_path, img['file_name']))
                for k, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    filename = f"{img['file_name'][:-4]}_{k}.jpg"
                    imwrite(image[y1:y2 + 1, x1:x2 + 1, :], os.path.join(self.gen_path, filename))
                    img_shapes[filename] = (y2 - y1 + 1, x2 - x1 + 1)

        pickle.dump(img_shapes, open(os.path.join(self.gen_path, 'shape.pkl'), 'wb'))


class AllMixDatasetBuilder:

    def __init__(self, json_path, dataset_path, gen_json_path, gen_path, gen_vis_path, gen_bg_path,
                 paste_n=10, fail_times=50, resize_times=10, brightness_diff=30, seg_mask_margin=5, use_seg=True,
                 paste_min_size=10, use_poisson=True, require_connect=False, sample_same_cls=False,
                 add_bg=False, bg_div_fg=1, allow_bg_resize=True, use_same_cls_set=False):
        self.dataset_path = dataset_path
        self.gen_json_path = gen_json_path
        self.gen_path = gen_path
        self.gen_vis_path = gen_vis_path
        self.gen_bg_path = gen_bg_path

        self.paste_n = paste_n
        self.fail_times = fail_times
        self.resize_times = resize_times
        self.brightness_diff = brightness_diff
        self.seg_mask_margin = seg_mask_margin
        self.use_seg = use_seg
        self.paste_min_size = paste_min_size
        self.use_poisson = use_poisson
        self.require_connect = require_connect
        self.sample_same_cls = sample_same_cls
        self.add_bg = add_bg
        self.bg_div_fg = bg_div_fg
        self.allow_bg_resize = allow_bg_resize
        self.use_same_cls_set = use_same_cls_set
        self.bg_anns = None
        if self.add_bg:
            self.bg_anns = pickle.load(open(os.path.join(self.gen_bg_path, 'shape.pkl'), 'rb'))

        if not os.path.isdir(self.gen_path):
            os.makedirs(self.gen_path)
        if not os.path.isdir(self.gen_vis_path):
            os.makedirs(self.gen_vis_path)

        self.json_file = json.load(open(json_path, 'r'))
        self.coco = COCO(json_path)
        self.images = self.json_file['images']
        self.raw_annotations = self.json_file['annotations']
        self.annotations = [ann for ann in self.raw_annotations if ann['iscrowd'] == 0]
        self.crowd_annotations = [ann for ann in self.raw_annotations if ann['iscrowd'] == 1]
        self.cls_anns = defaultdict(list)
        for ann in self.annotations:
            self.cls_anns[ann['category_id']].append(ann)

        self.image_ids = [img['id'] for img in self.images]
        self.ann_ids = [ann['id'] for ann in self.annotations]
        self.image_id_to_idx = dict(zip(self.image_ids, range(len(self.image_ids))))
        self.image_id_to_ann_idxs = defaultdict(list)

        for idx, ann in enumerate(self.annotations):
            self.image_id_to_ann_idxs[ann['image_id']].append(idx)

        print(f"Number of images: {len(self.images)}")
        print(f"Number of raw bboxes: {len(self.raw_annotations)}")
        print(f"Number of bboxes: {len(self.annotations)}")

    def test_self_crop(self):
        img = self.images[self.image_id_to_idx[241196]]
        pasted_img, pasted_anns = self.paste_single(img, [
            self.annotations[ann_idx] for ann_idx in self.image_id_to_ann_idxs[241196]])

        bbox_result = [np.array(decode_bbox(ann['bbox'])) for ann in pasted_anns]
        label_result = [ann['category_id'] for ann in pasted_anns]

        show_result(pasted_img, [bbox_result, label_result],
                    out_file='/home/liuzili/drive/a.jpg')

    def test(self):
        img = self.images[4]
        for i in tqdm(range(20)):
            pasted_img, pasted_anns = self.paste_single(img)

            bbox_result = [np.array(decode_bbox(ann['bbox'])) for ann in pasted_anns]
            label_result = [ann['category_id'] for ann in pasted_anns]

            show_result(pasted_img, [bbox_result, label_result],
                        out_file=f'/home/liuzili/drive/test/a{i}.jpg')

    def mp(self, core_num, func, dargs):
        pool = Pool(core_num)
        res = []
        for args in dargs:
            res.append(pool.apply_async(func, args=(args,)))

        pool.close()
        pool.join()
        res = [r.get() for r in res]
        return res

    def gen_dataset_single(self, imgs):
        all_anns = []
        for img in tqdm(imgs):
            pasted_img, pasted_anns = self.paste_single(img)
            imwrite(pasted_img, f"{os.path.join(self.gen_path, img['file_name'])}")

            # bbox_result = [np.array(decode_bbox(ann['bbox'])) for ann in pasted_anns]
            # label_result = [ann['category_id'] for ann in pasted_anns]
            # show_result(pasted_img, [bbox_result, label_result],
            #             out_file=f"{os.path.join(self.gen_vis_path, img['file_name'])}")

            all_anns.extend(pasted_anns)
        return all_anns

    def gen_dataset(self, core_num=8):
        part = len(self.images) // core_num
        imgs = [self.images[i * part:(i + 1) * part] for i in range(core_num)]
        imgs[0].extend(self.images[core_num * part:])
        res = self.mp(core_num, func=self.gen_dataset_single, dargs=imgs)
        pasted_ann = []
        for r in res:
            pasted_ann.extend(r)
        pasted_ann.extend(self.crowd_annotations)
        gen_json_file = deepcopy(self.json_file)
        gen_json_file['annotations'] = pasted_ann
        with open(self.gen_json_path, 'w') as f:
            f.write(json.dumps(gen_json_file))

    def load_target_img(self, img):
        target_img = mmcv.imread(os.path.join(self.dataset_path, img['file_name']))
        target_anns = []
        target_mask = np.zeros_like(target_img[..., 0], dtype=np.bool)
        for ann_idx in self.image_id_to_ann_idxs[img['id']]:
            ann = self.annotations[ann_idx]
            target_anns.append(ann)
            if self.use_seg:
                fg_mask = self.coco.annToMask(ann).astype(np.bool)
                target_mask[fg_mask == 1] = 1
            else:
                x1, y1, x2, y2 = decode_bbox(ann['bbox'])
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                target_mask[y1:y2 + 1, x1:x2 + 1] = 1

        return target_img, target_anns, target_mask

    def is_connect(self, mask):
        small_mask = cv2.resize(mask.astype(np.uint8), (10, 10))
        if small_mask.sum() < 40:
            return False
        return True

    def load_bg_crops(self, masks):
        bg_crops = []
        bg_masks = []
        if len(masks) == 0:
            for _ in range(self.bg_div_fg):
                size_h = random.randint(20, 60)
                size_w = random.randint(20, 60)
                masks.append(np.ones((size_h, size_w), dtype=np.bool))
        for mask in masks:
            if mask is None:
                continue
            bg_ann_keys = random_select(list(self.bg_anns.keys()), self.bg_div_fg)
            for bg_ann_key in bg_ann_keys:
                bg_h, bg_w = self.bg_anns[bg_ann_key]
                mask_h, mask_w = mask.shape
                if bg_h > mask_h and bg_w > mask_w:
                    off_x = random.randint(0, bg_w - mask_w - 1)
                    off_y = random.randint(0, bg_h - mask_h - 1)
                    crop = mmcv.imread(os.path.join(self.gen_bg_path, bg_ann_key))[
                           off_y:off_y + mask_h, off_x:off_x + mask_w, :]
                    if crop.shape[:2] != mask.shape:
                        continue
                    bg_crops.append(crop)
                    bg_masks.append(mask)
                elif self.allow_bg_resize:
                    bg_crops.append(cv2.resize(mmcv.imread(
                        os.path.join(self.gen_bg_path, bg_ann_key)),
                        (mask.shape[1], mask.shape[0])))
                    bg_masks.append(mask)

        return bg_crops, bg_masks

    def load_source_crops(self, anns):
        source_crops = []
        source_masks = []
        bg_crops = []
        bg_masks = []
        for ann in anns:
            x1, y1, x2, y2 = decode_bbox(ann['bbox'])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if self.use_seg:
                mask = self.coco.annToMask(ann)[y1:y2 + 1, x1:x2 + 1].astype(np.bool)
            else:
                mask = np.ones((y2 - y1 + 1, x2 - x1 + 1), dtype=np.bool)
            if (x2 - x1 > self.paste_min_size and y2 - y1 > self.paste_min_size) and (
                    not self.require_connect or self.is_connect(mask)):
                source_img_id = ann['image_id']
                source_img_path = os.path.join(
                    self.dataset_path,
                    self.images[self.image_id_to_idx[source_img_id]]['file_name'])
                source_crops.append(mmcv.imread(source_img_path)[y1:y2 + 1, x1:x2 + 1, :])
                source_masks.append(mask)
            else:
                source_crops.append(None)
                source_masks.append(None)

        if self.add_bg:
            bg_crops, bg_masks = self.load_bg_crops(source_masks)

        return source_crops, source_masks, bg_crops, bg_masks

    def update_ann_bbox(self, ann, off):
        # TODO ann idx unique? modify seg?
        assert isinstance(off, (tuple, list))
        off_x, off_y = off
        new_ann = deepcopy(ann)
        x1, y1, x2, y2 = decode_bbox(new_ann['bbox'])
        x1, y1, x2, y2 = off_x, off_y, off_x + x2 - x1, off_y + y2 - y1
        randint = random.randint(1, 909000292289)
        while randint in self.ann_ids:
            randint = random.randint(1, 909000292289)
        self.ann_ids.append(randint)
        new_ann['id'] = randint
        new_ann['bbox'] = encode_bbox([x1, y1, x2, y2])
        return new_ann

    def paste_single_single(self, target_img, target_mask, crop, mask, ann=None):
        is_success = False
        target_h, target_w = target_img.shape[:2]
        source_h, source_w = crop.shape[:2]
        if target_h > source_h > self.paste_min_size and target_w > source_w > self.paste_min_size:
            off_x = random.randint(0, target_w - source_w)
            off_y = random.randint(0, target_h - source_h)
            target_sub_mask = target_mask[off_y:off_y + source_h, off_x:off_x + source_w]

            crop_bri = ImageStat.Stat(
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).mean[0]
            target_sub_bri = ImageStat.Stat(
                Image.fromarray(cv2.cvtColor(
                    target_img[off_y:off_y + source_h, off_x:off_x + source_w],
                    cv2.COLOR_BGR2RGB))).mean[0]
            if (self.brightness_diff > crop_bri - target_sub_bri > -self.brightness_diff) and \
                    not ((target_sub_mask == 1) & (mask == 1)).any():
                if ann is not None:
                    ann = self.update_ann_bbox(ann, (off_x, off_y))
                target_sub_mask[mask == 1] = 1  # update target_mask

                if self.seg_mask_margin > 0:
                    mask_torch = torch.tensor(mask).float()[None, None]
                    mask = F.max_pool2d(mask_torch, self.seg_mask_margin * 2 + 1,
                                        stride=1, padding=self.seg_mask_margin
                                        ).squeeze(1).squeeze(0).bool().numpy()

                if self.use_poisson:
                    target_img = cv2.seamlessClone(
                        crop, target_img, mask.astype(np.uint8) * 255,
                        (off_x + source_w // 2, off_y + source_h // 2), cv2.NORMAL_CLONE)
                else:
                    target_img[off_y:off_y + source_h, off_x:off_x + source_w][mask] = crop[mask]
                is_success = True

        return is_success, target_img, target_mask, ann

    def random_select_anns(self, target_anns):
        if self.sample_same_cls:
            target_clses = [ann['category_id'] for ann in target_anns]
            if self.use_same_cls_set:
                target_clses = list(set(target_clses))
            anns = []
            for cls in target_clses:
                anns.extend(random_select(self.cls_anns[cls], self.paste_n))
        else:
            anns = random_select(self.annotations, self.paste_n)
        return anns

    def paste_single_loop(self, target_img, target_mask, crop, mask, target_anns=None, ann=None):
        if crop is None or mask is None:
            return target_img, target_mask, target_anns

        ann_cp = None
        if ann is not None:
            ann_cp = deepcopy(ann)

        i = 0
        is_success = False
        ann_new = None
        while not is_success and i < self.fail_times:
            i += 1
            if i % self.resize_times == 0:
                source_h, source_w = crop.shape[0] // 2, crop.shape[1] // 2
                if source_w <= self.paste_min_size or source_h <= self.paste_min_size:
                    break
                crop = cv2.resize(crop, (source_w, source_h))
                mask = cv2.resize(mask.astype(crop.dtype), (source_w, source_h)).astype(
                    np.bool)
                if ann is not None:
                    ann_cp['bbox'] = encode_bbox([0, 0, source_w, source_h])

            is_success, target_img, target_mask, ann_new = self.paste_single_single(
                target_img, target_mask, crop, mask, ann_cp)

        if is_success and target_anns is not None:
            target_anns.append(ann_new)
        return target_img, target_mask, target_anns

    def paste_single(self, img, anns=None):
        # paste single image
        target_img, target_anns, target_mask = self.load_target_img(img)
        if anns is not None:
            assert isinstance(anns, (tuple, list))
        else:
            anns = self.random_select_anns(target_anns)

        source_crops, source_masks, bg_crops, bg_masks = self.load_source_crops(anns)

        ann = None
        for source_crop, source_mask, source_ann in zip(source_crops, source_masks, anns):
            # target_img, target_mask, target_anns = self.paste_single_loop(
            #     target_img, target_mask, source_crop, source_mask, target_anns, source_ann)

            crop = source_crop
            mask = source_mask
            if crop is None or mask is None:
                continue
            ann = deepcopy(source_ann)
            i = 0
            is_success = False
            while not is_success and i < self.fail_times:
                i += 1
                if i % self.resize_times == 0:
                    source_h, source_w = crop.shape[0] // 2, crop.shape[1] // 2
                    if source_w > self.paste_min_size and source_h > self.paste_min_size:
                        crop = cv2.resize(crop, (source_w, source_h))
                        mask = cv2.resize(mask.astype(crop.dtype), (source_w, source_h)).astype(
                            np.bool)
                        ann['bbox'] = encode_bbox([0, 0, source_w, source_h])
                    else:
                        break

                try:
                    is_success, target_img, ann_new = self.paste_single_single(
                        target_img, target_mask, crop, mask, ann)
                except:
                    is_success = False

            if is_success:
                target_anns.append(ann_new)

        for bg_crop, bg_mask in zip(bg_crops, bg_masks):
            crop = bg_crop
            mask = bg_mask
            i = 0
            is_success = False
            while not is_success and i < self.fail_times:
                i += 1
                if i % self.resize_times == 0:
                    source_h, source_w = crop.shape[0] // 2, crop.shape[1] // 2
                    if source_w > self.paste_min_size and source_h > self.paste_min_size:
                        crop = cv2.resize(crop, (source_w, source_h))
                        mask = cv2.resize(mask.astype(crop.dtype), (source_w, source_h)).astype(
                            np.bool)
                    else:
                        break
                try:
                    is_success, target_img, _ = self.paste_single_single(
                        target_img, target_mask, crop, mask, ann)
                except:
                    is_success = False

        return target_img, target_anns


def show_result(img, result, class_names=None, score_thr=0., out_file=None):
    bbox_result, label_result = result
    bboxes = np.vstack(bbox_result)
    labels = np.array(label_result)
    imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None,
        out_file=out_file)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, bboxes.shape[1]
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def get_params():
    return {
        '1': dict(),
        '3': dict(use_poisson=False),
        '4': dict(paste_n=5, brightness_diff=20),
        '5': dict(paste_n=5, brightness_diff=20, use_poisson=False),  # 6.2
        '6': dict(paste_n=5, brightness_diff=20, paste_min_size=50, use_poisson=False),  # 8.3
        '7': dict(paste_n=5, brightness_diff=20, paste_min_size=30, use_poisson=False),
        '8': dict(paste_n=5, brightness_diff=20, paste_min_size=50, use_poisson=False,  # 8.3
                  require_connect=True),
        '9': dict(paste_n=10, brightness_diff=20, paste_min_size=50, use_poisson=True,
                  require_connect=True),
        # 10, 11
        '10': dict(paste_n=2, brightness_diff=20, paste_min_size=30, use_poisson=False,  # 8.3
                   require_connect=True, sample_same_cls=True),
        '11': dict(paste_n=2, brightness_diff=20, paste_min_size=30, use_poisson=True,  # 8.2
                   require_connect=True, sample_same_cls=True),

        '12': dict(paste_n=5, brightness_diff=20, paste_min_size=30, use_poisson=True,
                   # 7.8 / 8.6(2hm) / 5.3(2lr)
                   require_connect=True, sample_same_cls=True, seg_mask_margin=7),
        '13': dict(paste_n=5, brightness_diff=20, paste_min_size=30, use_poisson=True,  # 0.3
                   require_connect=True, sample_same_cls=True, seg_mask_margin=1),
        '14': dict(paste_n=3, brightness_diff=20, paste_min_size=30, use_poisson=True,  # 8.6
                   require_connect=True, sample_same_cls=True, add_bg=True),

        # 15, 16, 17
        '15': dict(paste_n=0, brightness_diff=20, paste_min_size=30, use_poisson=True,  #
                   require_connect=True, sample_same_cls=True, add_bg=True),
        '16': dict(paste_n=0, brightness_diff=20, paste_min_size=30, use_poisson=True,  #
                   require_connect=True, sample_same_cls=True, add_bg=True, bg_div_fg=3),
        '17': dict(paste_n=0, brightness_diff=20, paste_min_size=30, use_poisson=False,  #
                   require_connect=True, sample_same_cls=True, add_bg=True, bg_div_fg=3),

        '18': dict(paste_n=2, brightness_diff=20, paste_min_size=30, use_poisson=False,  #
                   require_connect=True, sample_same_cls=True, add_bg=True),
        '19': dict(paste_n=2, brightness_diff=20, paste_min_size=30, use_poisson=True,  #
                   require_connect=True, sample_same_cls=True, add_bg=True),

        '20': dict(paste_n=2, brightness_diff=100, paste_min_size=10, use_poisson=True,  #
                   require_connect=False, sample_same_cls=False, seg_mask_margin=0, add_bg=True),
        '25': dict(paste_n=2, brightness_diff=100, paste_min_size=10, use_poisson=True,  #
                   require_connect=False, sample_same_cls=False, seg_mask_margin=1, add_bg=True),
        '21': dict(paste_n=2, brightness_diff=20, paste_min_size=10, use_poisson=True,  #
                   require_connect=False, sample_same_cls=False, seg_mask_margin=1, add_bg=True),
        '22': dict(paste_n=2, brightness_diff=100, paste_min_size=10, use_poisson=True,  #
                   require_connect=False, sample_same_cls=False, seg_mask_margin=3, add_bg=True),
        '23': dict(paste_n=2, brightness_diff=100, paste_min_size=10, use_poisson=False,  #
                   require_connect=False, sample_same_cls=False, seg_mask_margin=0, add_bg=True),
        '24': dict(paste_n=2, brightness_diff=100, paste_min_size=10, use_poisson=False,  #
                   require_connect=False, sample_same_cls=True, seg_mask_margin=0, add_bg=True),
    }


if __name__ == '__main__':
    test_on_mine = False

    if os.environ['USER'] != 'feiyuejiao':
        version = '21'
        if socket.gethostname() == 'gpu9.fabu.ai':
            drive = 'drive9'
        else:
            drive = 'drive7'
        json_path = f'/home/liuzili/{drive}/data/coco/annotations/instances_train2017_small.json'
        dataset_path = f'/home/liuzili/{drive}/data/coco/train2017_small'
        gen_bg_path = f'/home/liuzili/{drive}/data/coco/train2017_bg'
        gen_json_path = f'/home/liuzili/{drive}/data/coco/annotations/instances_train2017_small_gen{version}.json'
        gen_path = f'/home/liuzili/{drive}/data/coco/train2017_small_gen{version}'
        gen_vis_path = f'/home/liuzili/{drive}/data/coco/train2017_small_gen{version}_vis'

        # m = BGMemoryBuilder(json_path, dataset_path, gen_bg_path)
        # m.gen_bg()
    else:
        version = '14'
        json_path = '/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/annotations/instances_train2017_small.json'
        dataset_path = '/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/train2017_small'
        gen_bg_path = '/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/train2017_bg'
        # m = BGMemoryBuilder(json_path, dataset_path, gen_bg_path)
        # m.gen_bg()
        gen_json_path = f'/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/annotations/instances_train2017_small_gen{version}.json'
        gen_path = f'/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/train2017_small_gen{version}'
        gen_vis_path = f'/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/train2017_small_gen{version}_vis'

    print(f"Version: {version}")
    time.sleep(1)
    if test_on_mine and os.environ['USER'] != 'feiyuejiao':
        params = get_params()
        dataset = AllMixDatasetBuilder(
            json_path, dataset_path, gen_json_path, gen_path, gen_vis_path, gen_bg_path,
            **params[version])
        dataset.test()
    else:
        params = get_params()
        dataset = AllMixDatasetBuilder(
            json_path, dataset_path, gen_json_path, gen_path, gen_vis_path, gen_bg_path,
            **params[version])
        dataset.gen_dataset(core_num=38)
