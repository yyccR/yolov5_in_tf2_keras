import sys

sys.path.append("../../detector_in_keras")

import os
import cv2
import random
import re
import traceback
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io


class CoCoDataGenrator:
    def __init__(self,
                 coco_annotation_file,
                 train_img_nums=-1,
                 img_shape=(640, 640, 3),
                 batch_size=1,
                 max_instances=100,
                 include_crowd=False,
                 include_mask=False,
                 include_keypoint=False,
                 need_down_image=True,
                 download_image_path=os.path.dirname(os.path.abspath(__file__)) + "/" + './coco_2017_val_images/',
                 ):
        # 设置要训练的图片数, -1表示全部
        self.train_img_nums = train_img_nums
        # 是否需要下载图片数据, 只有官方CoCo数据才需要下载, 自己打标转CoCo格式不需要
        self.need_down_image = need_down_image
        # 设置下载保存coco json文件中图片的目录
        self.download_image_path = download_image_path
        # 图片最终resize+padding后的大小
        self.img_shape = img_shape
        self.batch_size = batch_size
        # 此参数为保证不同size的box,mask能padding到一个batch里
        self.max_instances = max_instances
        # 是否输出包含crowd类型数据
        self.include_crowd = include_crowd
        # 是否输出包含mask分割数据
        self.include_mask = include_mask
        # 是否输出包含keypoint数据
        self.include_keypoint = include_keypoint
        self.coco_annotation_file = coco_annotation_file

        self.current_batch_index = 0
        self.total_batch_size = 0
        self.img_ids = []
        self.coco = COCO(annotation_file=coco_annotation_file)
        self.load_data()
        if self.need_down_image:
            self.download_image_files()

    def load_data(self):
        # 初步过滤数据是否包含crowd
        target_img_ids = []
        for k in self.coco.imgToAnns:
            annos = self.coco.imgToAnns[k]
            if annos:
                annos = list(filter(lambda x: x['iscrowd'] == self.include_crowd, annos))
                if annos:
                    target_img_ids.append(k)

        if self.train_img_nums > 0:
            # np.random.shuffle(target_img_ids)
            target_img_ids = target_img_ids[:self.train_img_nums]

        self.total_batch_size = len(target_img_ids) // self.batch_size
        self.img_ids = target_img_ids

    def download_image_files(self):
        """下载coco图片数据"""
        if not os.path.exists(self.download_image_path):
            os.makedirs(self.download_image_path)

        if len(os.listdir(self.download_image_path)) > 0:
            print("image files already downloaded! size: {}".format(len(os.listdir(self.download_image_path))))

        for i, img_id in enumerate(self.img_ids):
            file_path = self.download_image_path + "./{}.jpg".format(img_id)
            if os.path.isfile(file_path):
                print("already exist file: {}".format(file_path))
            else:
                if self.coco.imgs[img_id].get("coco_url"):
                    try:
                        im = io.imread(self.coco.imgs[img_id]['coco_url'])
                        io.imsave(file_path, im)
                        print("save image {}, {}/{}".format(file_path, i+1, len(self.img_ids)))
                    except Exception as e:
                        traceback.print_exc()
                        print("current img_id: ", img_id, "current img_file: ", file_path)

    def next_batch(self):
        if self.current_batch_index >= self.total_batch_size:
            self.current_batch_index = 0
            self._on_epoch_end()

        batch_img_ids = self.img_ids[self.current_batch_index * self.batch_size:
                                     (self.current_batch_index + 1) * self.batch_size]
        batch_imgs = []
        batch_bboxes = []
        batch_labels = []
        batch_masks = []
        batch_keypoints = []
        valid_nums = []
        for img_id in batch_img_ids:
            # {"img":, "bboxes":, "labels":, "masks":, "key_points":}
            data = self._data_generation(image_id=img_id)
            if len(np.shape(data['imgs'])) > 0:
                batch_imgs.append(data['imgs'])
                batch_labels.append(data['labels'])
                batch_bboxes.append(data['bboxes'])
                valid_nums.append(data['valid_nums'])
                # if len(data['labels']) > self.max_instances:
                #     batch_bboxes.append(data['bboxes'][:self.max_instances, :])
                #     batch_labels.append(data['labels'][:self.max_instances])
                #     valid_nums.append(self.max_instances)
                # else:
                #     pad_num = self.max_instances - len(data['labels'])
                #     batch_bboxes.append(np.pad(data['bboxes'], [(0, pad_num), (0, 0)]))
                #     batch_labels.append(np.pad(data['labels'], [(0, pad_num)]))
                #     valid_nums.append(len(data['labels']))

                if self.include_mask:
                    batch_masks.append(data['masks'])

                if self.include_keypoint:
                    batch_keypoints.append(data['keypoints'])

        self.current_batch_index += 1

        if len(batch_imgs) < self.batch_size:
            return self.next_batch()

        output = {
            'imgs': np.array(batch_imgs, dtype=np.int32),
            'bboxes': np.array(batch_bboxes, dtype=np.int16),
            'labels': np.array(batch_labels, dtype=np.int8),
            'masks': np.array(batch_masks, dtype=np.int8),
            'keypoints': np.array(batch_keypoints, dtype=np.int16),
            'valid_nums': np.array(valid_nums, dtype=np.int8)
        }

        return output

    def _on_epoch_end(self):
        np.random.shuffle(self.img_ids)

    def _resize_im(self, origin_im, bboxes):
        """ 对图片/mask/box resize

        :param origin_im
        :param bboxes
        :return im_blob: [h, w, 3]
                gt_boxes: [N, [ymin, xmin, ymax, xmax]]
        """
        im_shape = np.shape(origin_im)
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.img_shape[0]) / float(im_size_max)

        # resize原始图片
        im_resize = cv2.resize(origin_im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_resize_shape = np.shape(im_resize)
        im_blob = np.zeros(self.img_shape, dtype=np.float32)
        im_blob[0:im_resize_shape[0], 0:im_resize_shape[1], :] = im_resize

        # resize对应边框
        bboxes_resize = np.array(bboxes * im_scale, dtype=np.int16)

        return im_blob, bboxes_resize

    def _resize_mask(self, origin_masks):
        """ resize mask数据
        :param origin_mask:
        :return: mask_resize: [instance, h, w]
                 gt_boxes: [N, [ymin, xmin, ymax, xmax]]
        """
        mask_shape = np.shape(origin_masks)
        mask_size_max = np.max(mask_shape[0:3])
        im_scale = float(self.img_shape[0]) / float(mask_size_max)

        # resize mask/box
        gt_boxes = []
        masks_resize = []
        for m in origin_masks:
            m = np.array(m, dtype=np.float32)
            m_resize = cv2.resize(m, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            m_resize = np.array(m_resize >= 0.5, dtype=np.int8)

            # 计算bdbox
            h, w = np.shape(m_resize)
            rows, cols = np.where(m_resize)
            # [xmin, ymin, xmax, ymax]
            xmin = np.min(cols) if np.min(cols) >= 0 else 0
            ymin = np.min(rows) if np.min(rows) >= 0 else 0
            xmax = np.max(cols) if np.max(cols) <= w else w
            ymax = np.max(rows) if np.max(rows) <= h else h
            bdbox = [xmin, ymin, xmax, ymax]
            gt_boxes.append(bdbox)

            mask_blob = np.zeros((self.img_shape[0], self.img_shape[1], 1), dtype=np.float32)
            mask_blob[0:h, 0:w, 0] = m_resize
            masks_resize.append(mask_blob)

        # [instance_num, [xmin, ymin, xmax, ymax]]
        gt_boxes = np.array(gt_boxes, dtype=np.int16)
        # [h, w, instance_num]
        masks_resize = np.concatenate(masks_resize, axis=-1)

        return masks_resize, gt_boxes

    def _data_generation(self, image_id):
        """ 拉取coco标记数据, 目标边框, 类别, mask
        :param image_id:
        :return:
        """
        anno_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=self.include_crowd)
        bboxes = []
        labels = []
        masks = []
        keypoints = []
        for i in anno_ids:
            # 边框, 处理成左上右下坐标
            ann = self.coco.anns[i]
            bbox = ann['bbox']
            xmin, ymin, w, h = bbox
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            bboxes.append([xmin, ymin, xmax, ymax])
            # 类别ID
            label = ann['category_id']
            labels.append(label)
            # 实例分割
            if self.include_mask:
                # [instances, h, w]
                mask = self.coco.annToMask(ann)
                masks.append(mask)
            if self.include_keypoint and ann.get('keypoints'):
                keypoint = ann['keypoints']
                # 处理成[x,y,v] 其中v=0表示没有此点,v=1表示被挡不可见,v=2表示可见
                keypoint = np.reshape(keypoint, [-1, 3])
                keypoints.append(keypoint)

        # 输出包含5个东西, 不需要则为空
        outputs = {
            "imgs": [],
            "labels": [],
            "bboxes": [],
            "masks": [],
            "keypoints": [],
            "valid_nums": 0
        }

        valid_nums = 0
        if len(labels) > self.max_instances:
            bboxes = bboxes[:self.max_instances, :]
            labels = labels[:self.max_instances]
            valid_nums = self.max_instances
            # batch_bboxes.append(data['bboxes'][:self.max_instances, :])
            # batch_labels.append(data['labels'][:self.max_instances])
            # valid_nums.append(self.max_instances)
        else:
            pad_num = self.max_instances - len(labels)
            bboxes = np.pad(bboxes, [(0, pad_num), (0, 0)])
            labels = np.pad(labels, [(0, pad_num)])
            valid_nums = self.max_instances - pad_num
            # batch_bboxes.append(np.pad(data['bboxes'], [(0, pad_num), (0, 0)]))
            # batch_labels.append(np.pad(data['labels'], [(0, pad_num)]))
            # valid_nums.append(len(data['labels']))

        # 处理最终数据 mask
        if self.include_mask:
            # [h, w, instances]
            masks, _ = self._resize_mask(origin_masks=masks)
            if np.shape(masks)[2] > self.max_instances:
                masks = masks[:self.max_instances, :, :]
            else:
                pad_num = self.max_instances - np.shape(masks)[2]
                masks = np.pad(masks, [(0, 0), (0, 0), (0, pad_num)])

            outputs['masks'] = masks
            # outputs['bboxes'] = bboxes

        # 处理最终数据 keypoint
        if self.include_keypoint:
            keypoints = np.array(keypoints, dtype=np.int8)
            outputs['keypoints'] = keypoints

        # img = io.imread(self.coco.imgs[image_id]['coco_url'])
        # img_file = self.download_image_path + "./{}.jpg".format(image_id)
        # if not os.path.isfile(img_file):
        #     file_path = self.download_image_path + "./{}.jpg".format(image_id)
        #     print("download image from {}".format(self.coco.imgs[image_id]['coco_url']))
        #     im = io.imread(self.coco.imgs[image_id]['coco_url'])
        #     io.imsave(file_path, im)
        #     print("save image {}".format(file_path))
        # img = cv2.imread(img_file)
        img_coco_url_file = str(self.coco.imgs[image_id].get('coco_url', ""))
        img_url_file = str(self.coco.imgs[image_id].get('url', ""))
        img_local_file = str(self.coco.imgs[image_id].get('file_name', "")).encode('unicode_escape').decode()
        img_local_file = os.path.join(os.path.dirname(self.coco_annotation_file), img_local_file)
        img_local_file = re.sub(r"\\\\", "/", img_local_file)

        img = []

        if os.path.isfile(img_local_file):
            img = io.imread(img_local_file)
        elif img_coco_url_file.startswith("http"):
            download_image_file = self.download_image_path + "./{}.jpg".format(image_id)
            if not os.path.isfile(download_image_file):
                print("download image from {}".format(img_coco_url_file))
                im = io.imread(img_coco_url_file)
                io.imsave(download_image_file, im)
                print("save image {}".format(download_image_file))
            img = io.imread(download_image_file)
        elif img_url_file.startswith("http"):
            download_image_file = self.download_image_path + "./{}.jpg".format(image_id)
            if not os.path.isfile(download_image_file):
                print("download image from {}".format(img_url_file))
                im = io.imread(img_url_file)
                io.imsave(download_image_file, im)
                print("save image {}".format(download_image_file))
            img = io.imread(download_image_file)
        else:
            return outputs

        if len(np.shape(img)) < 2:
            return outputs
        elif len(np.shape(img)) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # img = np.expand_dims(img, axis=-1)
            # img = np.pad(img, [(0, 0), (0, 0), (0, 2)])
        else:
            img = img[:, :, ::-1]

        labels = np.array(labels, dtype=np.int8)
        bboxes = np.array(bboxes, dtype=np.int16)
        img_resize, bboxes_resize = self._resize_im(origin_im=img, bboxes=bboxes)

        outputs['imgs'] = img_resize
        outputs['labels'] = labels
        outputs['bboxes'] = bboxes_resize
        outputs['valid_nums'] = valid_nums

        return outputs


if __name__ == "__main__":
    from data.visual_ops import draw_bounding_box, draw_instance

    file = "./cat_dog_face_data/train_annotations.json"
    # file = "./instances_val2017.json"
    # file = "./yanhua/annotations.json"
    coco = CoCoDataGenrator(
        coco_annotation_file=file,
        train_img_nums=8,
        include_mask=False,
        include_keypoint=False,
        need_down_image=False,
        batch_size=8)

    # data = coco.next_batch()
    data = coco._data_generation(2)
    gt_imgs = data['imgs']
    gt_boxes = data['bboxes']
    gt_classes = data['labels']
    gt_masks = data['masks']
    valid_nums = data['valid_nums']

    img = gt_imgs if len(np.shape(gt_imgs)) == 3 else gt_imgs[-1]
    valid_num = [valid_nums] if type(valid_nums) == int else valid_nums
    gt_classes = [gt_classes] if len(np.shape(gt_classes)) == 1 else gt_classes
    gt_boxes = [gt_boxes] if len(np.shape(gt_boxes)) == 2 else gt_boxes
    gt_masks = [gt_masks] if len(np.shape(gt_masks)) == 3 else gt_masks
    for i in range(valid_num[-1]):
        label = float(gt_classes[-1][i])
        label_name = coco.coco.cats[label]['name']
        x1, y1, x2, y2 = gt_boxes[-1][i]
        # mask = gt_masks[-1][:, :, i]
        # img = draw_instance(img, mask)
        img = draw_bounding_box(img, label_name, label, x1, y1, x2, y2)
    cv2.imshow("", img)
    cv2.waitKey(0)

    # data = coco.next_batch()
    # print(data)
    # for i in range(90):
    #     if coco.coco.cats.get(i):
    #         print(coco.coco.cats[i]['name'])
    #     else:
    #         print("none")
    #
    # for i in coco.coco.cats:
    #     print(i)
    # outputs = coco._data_generation(image_id=348045)
    # cv2.imshow("image", np.array(outputs['img'],dtype=np.uint8))
    # cv2.waitKey(0)
