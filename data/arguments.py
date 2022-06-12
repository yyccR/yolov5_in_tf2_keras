import random
import math
import cv2
import numpy as np


class Arugments:
    def __init__(self,
                 image_shape=(640, 640, 3),
                 hsv_h=0.2,
                 hsv_s=0.7,
                 hsv_v=0.4,
                 degrees=30.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.3,
                 perspective=0.001,
                 mosaic_min_area_percent=0.4,
                 mix_up_beta=32.0):
        self.image_shape = image_shape
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mosaic_min_area_percent = mosaic_min_area_percent
        self.mix_up_beta = mix_up_beta

    def _xyxy_pad_and_clip(self, x, padw=0, padh=0, min_size=0, max_size=640):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = np.clip(x[:, 0] + padw, min_size, max_size)  # top left x
        y[:, 1] = np.clip(x[:, 1] + padh, min_size, max_size)  # top left y
        y[:, 2] = np.clip(x[:, 2] + padw, min_size, max_size)  # bottom right x
        y[:, 3] = np.clip(x[:, 3] + padh, min_size, max_size)  # bottom right y
        return y

    def _bbox_iou(self, box1, box2, eps=1E-7):
        """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
        box1:       np.array of shape(4)
        box2:       np.array of shape(nx4)
        returns:    np.array of shape(n)
        """
        box2 = box2.transpose()
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps
        # Intersection over box2 area
        return inter_area / box2_area

    def _box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

    def random_mosaic(self, images, boxes, target_img_size):
        """ 4图拼接
        :param images: (4, , , c)
        :param boxes: (4, (x1,y1,x2,y2,...))
        :param img_size: 输出的图片大小
        :return: [h,w,c]
        """
        mosaic_border = [-target_img_size // 2, -target_img_size // 2]
        boxes4 = []
        s = target_img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y
        for i, img in enumerate(images):
            # Load image
            # img, _, (h, w) = self.load_image(index)
            h, w, _ = img.shape
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                # 这里计算第一张图贴到左上角部分的一个 起点xy, 终点xy就是xc,yc
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                # 计算主要是裁剪出要贴的图，避免越界了, 其实起点一般就是(0,0),如果上面xc<w,yc<h,这里就会被裁剪掉部分, 终点就是w,h
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # 处理每张图片对应的boxes
            if boxes[i].size:
                new_boxes = self._xyxy_pad_and_clip(boxes[i], padw, padh, 0, 2 * s)
                new_area = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
                origin_area = (boxes[i][:, 2] - boxes[i][:, 0]) * (boxes[i][:, 3] - boxes[i][:, 1])
                new_boxes = new_boxes[(new_area / origin_area) >= self.mosaic_min_area_percent, :]
                if new_boxes.shape[0]:
                    boxes4.append(new_boxes)

        if boxes4:
            boxes4 = np.concatenate(boxes4, axis=0)
        return img4, boxes4

    def random_perspective(self, im, boxes=(), border=(0, 0)):
        """ 随机映射变换, 此步包含平移, 透视变换, 旋转缩放, 错切, 平移"""
        # targets = [cls, xyxy]

        height = im.shape[0] + border[0] * 2  # shape(h,w,c)
        width = im.shape[1] + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # 综合所有变换矩阵
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        # 目标边框跟随变换
        n = len(boxes)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # filter candidates
            i = self._box_candidates(box1=boxes[:, :4].T * s, box2=new.T, area_thr=0.10)
            boxes = boxes[i]
            boxes[:, :4] = new[i]

        return im, boxes

    def random_hsv(self, im):
        """ 曝光, 饱和, 亮度变换 """
        im_hsv = im.copy()
        if self.hsv_h or self.hsv_s or self.hsv_v:
            r = np.random.uniform(-1, 1, 3) * [self.hsv_h, self.hsv_s, self.hsv_v] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            im_hsv = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return im_hsv

    def random_flip(self, im, boxes, flip_up=True, flip_right=True):
        """ 上下左右翻转 """
        nl = len(boxes)
        h, w, _ = im.shape
        new_boxes = np.copy(boxes)
        if flip_up:
            im = np.flipud(im)
            if nl:
                new_boxes[:, (1, 3)] = h - boxes[:, (1, 3)]
                new_boxes[:, :-1] = new_boxes[:, (0, 3, 2, 1)]
        if flip_right:
            im = np.fliplr(im)
            if nl:
                new_boxes[:, (0, 2)] = w - boxes[:, (0, 2)]
                new_boxes[:, :-1] = new_boxes[:, (2, 1, 0, 3)]
        return im, new_boxes

    def random_mixup(self, im, boxes, im2, boxes2):
        """图像融合 https://arxiv.org/pdf/1710.09412.pdf
        :param im
        :param boxes: [n, (x1,y1,x2,y2,...)]
        :param im2
        :param boxes2: [m, (x1,y1,x2,y2,...)]
        """
        r = np.random.beta(self.mix_up_beta, self.mix_up_beta)
        im = (im * r + im2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((boxes, boxes2), 0)
        return im, labels

    def random_copy_paste(self, im_origin, masks_origin, boxes_origin, target_im, target_masks, target_boxes, p=1.):
        """ 随机分割填补 https://arxiv.org/abs/2012.07177
        :param im_origin:
        :param masks_origin:
        :param boxes_origin:
        :param target_im:
        :param target_masks:
        :param target_boxes:
        :param p:
        :return:
        """

        out_boxes = []
        out_masks = []
        n = target_masks.shape[-1]
        im_new = im_origin.copy()
        if p and n:
            h, w, c = im_origin.shape  # height, width, channels
            for j in random.sample(range(n), k=round(p * n)):
                start_x = np.random.uniform(0, w // 2)
                start_y = np.random.uniform(0, h // 2)
                box, mask = target_boxes[j], target_masks[:, :, j:j + 1]
                new_box = [
                    int(start_x),
                    int(start_y),
                    int(min(start_x + (box[2] - box[0]), w)),
                    int(min(start_y + (box[3] - box[1]), h)),
                ]
                iou = self._bbox_iou(new_box, boxes_origin)
                if (iou < 0.60).all():
                    mask_im = (target_im * mask)[
                              box[1]:int((new_box[3] - new_box[1]) + box[1]),
                              box[0]:int((new_box[2] - new_box[0])) + box[0], :]
                    new_mask_im = np.zeros(shape=(h, w, 3), dtype=int)
                    new_mask_im[new_box[1]:new_box[3], new_box[0]:new_box[2], :] = mask_im
                    # cv2.imshow("", np.array(new_mask_im, dtype=np.uint8))

                    target_mask = mask[
                                  box[1]:int((new_box[3] - new_box[1]) + box[1]),
                                  box[0]:int((new_box[2] - new_box[0])) + box[0], :]
                    new_mask = np.zeros(shape=(h, w, 1), dtype=int)
                    new_mask[new_box[1]:new_box[3], new_box[0]:new_box[2], :] = target_mask
                    out_masks.append(new_mask)
                    box[:4] = new_box
                    out_boxes.append(box)

                    im_new = im_new * (1 - new_mask) + new_mask_im * new_mask

        out_boxes = np.array(out_boxes)
        out_boxes = np.concatenate([boxes_origin, out_boxes], axis=0)
        out_masks = np.concatenate(out_masks, axis=-1)
        out_masks = np.concatenate([masks_origin, out_masks], axis=-1)
        im_new = np.array(im_new, dtype=np.uint8)
        return im_new, out_boxes, out_masks


if __name__ == "__main__":

    from data.visual_ops import draw_bounding_box, draw_instance
    from data.generate_coco_data import CoCoDataGenrator

    f = "./tmp/Cats_Test49.jpg"
    im = cv2.imread(f)
    box = np.array([[113, 137, 113 + 118, 137 + 139,1]])

    ag = Arugments()
    im = ag.random_hsv(im)
    im, box = ag.random_perspective(im, box)
    im2, box2 = ag.random_flip(im, box)
    im3, box3 = ag.random_mosaic(images=[im, im, im, im], boxes=([box, box, box, box]),
                                 target_img_size=320)
    im4, box4 = ag.random_mixup(im, box, im2, box2)
    coco = CoCoDataGenrator(
        coco_annotation_file="./instances_val2017.json",
        train_img_nums=8,
        include_mask=True,
        include_keypoint=False,
        batch_size=8)
    data = coco.next_batch()
    gt_imgs = data['imgs']
    gt_boxes = data['bboxes']
    gt_classes = data['labels']
    gt_masks = data['masks']
    valid_nums = data['valid_nums']
    im5, box5, masks5 = ag.random_copy_paste(
        im_origin=gt_imgs[-1],
        masks_origin=gt_masks[-1][:, :, :valid_nums[-1]],
        boxes_origin=gt_boxes[-1][:valid_nums[-1]],
        target_im=gt_imgs[-2],
        target_masks=gt_masks[-2][:, :, :valid_nums[-2]],
        target_boxes=gt_boxes[-2][:valid_nums[-2]])

    for i, b in enumerate(box5):
        im5 = draw_bounding_box(im5, "dog", 1, x_min=b[0], y_min=b[1], x_max=b[2], y_max=b[3])
        im5 = draw_instance(im5, masks5[:, :, i])
    cv2.imshow("", im5)
    cv2.waitKey(0)
