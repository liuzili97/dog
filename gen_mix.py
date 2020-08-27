import os
import json
import mmcv
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F
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


class AllMixDatasetBuilder:

    def __init__(self, json_path, dataset_path, gen_json_path, gen_path, gen_vis_path,
                 paste_n=10, fail_times=50, brightness_diff=30, seg_mask_margin=5,
                 paste_min_size=10, use_poisson=True, require_connect=False):
        self.dataset_path = dataset_path
        self.gen_json_path = gen_json_path
        self.gen_path = gen_path
        self.gen_vis_path = gen_vis_path
        self.paste_n = paste_n
        self.fail_times = fail_times
        self.brightness_diff = brightness_diff
        self.seg_mask_margin = seg_mask_margin
        self.paste_min_size = paste_min_size
        self.use_poisson = use_poisson
        self.require_connect = require_connect

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
        img = self.images[5]
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

    def gen_dataset(self):
        core_num = 16
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

            # x1, y1, x2, y2 = decode_bbox(ann['bbox'])
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # target_mask[y1:y2 + 1, x1:x2 + 1] = 1
            fg_mask = self.coco.annToMask(ann).astype(np.bool)
            target_mask[fg_mask == 1] = 1

        return target_img, target_anns, target_mask

    def is_connect(self, mask):
        small_mask = cv2.resize(mask.astype(np.uint8), (10, 10))
        if small_mask.sum() < 40:
            return False
        return True

    def load_source_crops(self, anns):
        source_crops = []
        source_masks = []
        for ann in anns:
            source_crops.append(None)
            source_masks.append(None)

            x1, y1, x2, y2 = decode_bbox(ann['bbox'])
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 - x1 > self.paste_min_size and y2 - y1 > self.paste_min_size:
                mask = self.coco.annToMask(ann)[y1:y2 + 1, x1:x2 + 1].astype(np.bool)
                if not self.require_connect or self.is_connect(mask):
                    source_img_id = ann['image_id']
                    source_img_path = os.path.join(
                        self.dataset_path,
                        self.images[self.image_id_to_idx[source_img_id]]['file_name'])
                    source_crops[-1] = mmcv.imread(source_img_path)[y1:y2 + 1, x1:x2 + 1, :]
                    # source_masks.append(np.ones_like(source_crops[-1][..., 0], dtype=np.bool))
                    source_masks[-1] = mask
        return source_crops, source_masks

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

    def paste_single_single(self, target_img, target_mask, crop, mask, ann):
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
                ann = self.update_ann_bbox(ann, (off_x, off_y))
                target_sub_mask[mask == 1] = 1  # update target_mask

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

        return is_success, target_img, ann

    def paste_single(self, img, anns=None):
        # paste single image
        if anns is not None:
            assert isinstance(anns, (tuple, list))
        else:
            rand_ids = list(range(len(self.annotations)))
            random.shuffle(rand_ids)
            anns = [self.annotations[i] for i in rand_ids[:self.paste_n]]

        target_img, target_anns, target_mask = self.load_target_img(img)
        source_crops, source_masks = self.load_source_crops(anns)

        for source_crop, source_mask, source_ann in zip(source_crops, source_masks, anns):
            crop = source_crop
            mask = source_mask
            if crop is None or mask is None:
                continue
            ann = deepcopy(source_ann)
            i = 0
            is_success = False
            while not is_success and i < self.fail_times:
                i += 1
                if i % 10 == 0:
                    source_h, source_w = crop.shape[0] // 2, crop.shape[1] // 2
                    if source_w > 10 and source_h > 10:
                        crop = cv2.resize(crop, (source_w, source_h))
                        mask = cv2.resize(mask.astype(crop.dtype), (source_w, source_h)).astype(
                            np.bool)
                        ann['bbox'] = encode_bbox([0, 0, source_w, source_h])

                is_success, target_img, ann_new = self.paste_single_single(
                    target_img, target_mask, crop, mask, ann)

            if is_success:
                target_anns.append(ann_new)
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
        '5': dict(paste_n=5, brightness_diff=20, use_poisson=False),
        '6': dict(paste_n=5, brightness_diff=20, paste_min_size=50, use_poisson=False),
        '7': dict(paste_n=5, brightness_diff=20, paste_min_size=30, use_poisson=False),
        '8': dict(paste_n=5, brightness_diff=20, paste_min_size=50, use_poisson=False,
                  require_connect=True),
        '9': dict(paste_n=10, brightness_diff=20, paste_min_size=50, use_poisson=True,
                  require_connect=True),
        '10': dict(paste_n=10, brightness_diff=20, paste_min_size=30, use_poisson=True,
                   require_connect=True),
    }


if __name__ == '__main__':
    if os.environ['USER'] != 'feiyuejiao':
        version = '10'
        params = get_params()
        json_path = '/home/liuzili/drive/data/coco/annotations/instances_train2017_small.json'
        dataset_path = '/home/liuzili/drive/data/coco/train2017_small'
        gen_json_path = f'/home/liuzili/drive/data/coco/annotations/instances_train2017_small_gen{version}.json'
        gen_path = f'/home/liuzili/drive/data/coco/train2017_small_gen{version}'
        gen_vis_path = f'/home/liuzili/drive/data/coco/train2017_small_gen{version}_vis'

        dataset = AllMixDatasetBuilder(
            json_path, dataset_path, gen_json_path, gen_path, gen_vis_path,
            **params[version])
        dataset.test()
    else:
        version = '9'
        params = get_params()
        json_path = '/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/annotations/instances_train2017_small.json'
        dataset_path = '/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/train2017_small'
        gen_json_path = f'/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/annotations/instances_train2017_small_gen{version}.json'
        gen_path = f'/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/train2017_small_gen{version}'
        gen_vis_path = f'/public/home/yuchangbingroup/feiyuejiao/fei2_workspace/coco/train2017_small_gen{version}_vis'

        dataset = AllMixDatasetBuilder(
            json_path, dataset_path, gen_json_path, gen_path, gen_vis_path,
            **params[version])
        dataset.gen_dataset()
