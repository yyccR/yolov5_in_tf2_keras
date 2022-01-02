import random
import math
import cv2
import numpy as np


def bbox_iou(box1, box2, eps=1E-7):
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


def rectangular(im, target_im_size=640, pad=0, stride=32):
    """ rectangular变换
    """
    h, w, _ = im.shape
    ratio = h / w
    if ratio < 1:
        shape = [ratio, 1]
    else:
        shape = [1, 1 / ratio]
    rect_shape = np.ceil(np.array(shape) * target_im_size / stride + pad).astype(int) * stride
    zeros = np.zeros([640, 640], dtype=int)

    # im_shape = np.shape(im)
    # im_size_max = np.max(im_shape[0:2])
    # im_scale = float(target_im_size) / float(im_size_max)
    #
    # # resize原始图片
    # im_resize = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    # im_resize_shape = np.shape(im_resize)
    #
    # pad_row_start = int((target_im_size - im_resize_shape[0]) / 2)
    # pad_col_start = int((target_im_size - im_resize_shape[1]) / 2)
    # im_blob = np.zeros([target_im_size, target_im_size, 3], dtype=np.float32) + 114
    #
    # im_blob[pad_row_start:im_resize_shape[0]+pad_row_start, pad_col_start:im_resize_shape[1]+pad_col_start, :] = im_resize
    # im_blob = np.array(im_blob[:,pad_col_start-20:im_resize_shape[1]+pad_col_start+20, :],dtype=np.uint8)
    # print(im_blob.shape)
    # cv2.imshow('', im_blob)
    # cv2.waitKey(0)

    # resize对应边框
    # bboxes_resize = np.array(bboxes * im_scale, dtype=np.int16)

    im2 = cv2.resize(im, rect_shape[::-1])
    print(im.shape, im2.shape)
    cv2.imshow("1", im)
    cv2.imshow("2", im2)
    cv2.waitKey(0)


def mosaic(im_size=640):
    """ 4图拼接 """
    mosaic_border = [-im_size // 2, -im_size // 2]
    labels4, segments4 = [], []
    s = im_size
    # 这里随机计算一个xy中心点
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)  # mosaic center x, y
    # indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    # random.shuffle(indices)
    im_files = [
        './tmp/000000000049.jpg',
        './tmp/000000000136.jpg',
        './tmp/000000000077.jpg',
        './tmp/000000000009.jpg',
    ]

    # mosaic 4张贴图的大小
    img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    for i, file in enumerate(im_files):
        # Load image
        # img, _, (h, w) = load_image(self, index)
        img = cv2.imread(file)
        h, w, _ = np.shape(img)

        # place img in img4
        if i == 0:  # top left
            # base image with 4 tiles
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

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

    cv2.imshow('', img4)
    cv2.waitKey(0)


def copy_paste(im_origin, boxes_origin, im_masks, masks, mask_boxes, p=1.):
    """ 分割填补,  https://arxiv.org/abs/2012.07177
    :param boxes_origin:  [[x1,y1,x2,y2], ....]
    :param masks: [h,w,instances]
    """

    out_boxes = []
    out_masks = []
    n = masks.shape[-1]
    im_new = im_origin.copy()
    if p and n:
        h, w, c = im_origin.shape  # height, width, channels
        for j in random.sample(range(n), k=round(p * n)):
            start_x = np.random.uniform(0, w // 2)
            start_y = np.random.uniform(0, h // 2)
            box, mask = mask_boxes[j], masks[:, :, j:j + 1]
            new_box = [
                int(start_x),
                int(start_y),
                int(min(start_x + (box[2] - box[0]), w)),
                int(min(start_y + (box[3] - box[1]), h))
            ]
            iou = bbox_iou(new_box, boxes_origin)
            if (iou < 0.90).all():
                mask_im = (im_masks * mask)[
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
                out_boxes.append(new_box)
                out_masks.append(new_mask)

                im_new = im_new * (1 - new_mask) + new_mask_im * new_mask

    out_boxes = np.array(out_boxes)
    out_masks = np.concatenate(out_masks, axis=-1)
    im_new = np.array(im_new, dtype=np.uint8)
    return im_new, out_boxes, out_masks


def perspective(im, p=0.001):
    """透视变换"""
    # pure python implement
    # h, w, c = im.shape
    # P = np.eye(3)
    # P[2, 0] = random.uniform(-p, p)
    # P[2, 1] = random.uniform(-p, p)
    # new_im = np.zeros_like(im) + 114.
    # for row in range(h):
    #     for col in range(w):
    #         col_new = (P[0, 0] * col + P[0, 1] * row + P[0, 2]) / (P[2, 0] * col + P[2, 1] * row + P[2, 2])
    #         row_new = (P[1, 0] * col + P[1, 1] * row + P[1, 2]) / (P[2, 0] * col + P[2, 1] * row + P[2, 2])
    #         new_im[int(row_new), int(col_new), :] = im[row, col, :]
    #
    # new_im = np.array(new_im, dtype=np.uint8)

    # yolov5 implement
    h, w, c = im.shape
    im_copy = im.copy()
    P = np.eye(3)
    P[2, 0] = random.uniform(-p, p)
    P[2, 1] = random.uniform(-p, p)
    new_im = cv2.warpPerspective(im_copy, P, dsize=(w, h), borderValue=(114, 114, 114))
    return new_im


def rotate_scale(im, degrees, scale):
    """旋转缩放"""
    im_copy = im.copy()
    h,w,_ = im.shape
    # Rotation and Scale matrix
    RS = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    random_scale = random.uniform(1 - scale, 1 + scale)
    RS[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=random_scale)
    print(RS)
    new_im = cv2.warpPerspective(im_copy, RS, dsize=(w, h), borderValue=(114, 114, 114))
    return new_im


def shear(im, degree):
    """错切"""
    im_copy = im.copy()
    h,w,_ = im.shape
    S = np.eye(3)
    # 错切和旋转都是通过[0,1],[1,0]两个参数控制, 不同的是旋转两个参数互为相反数, 错切则不然
    S[0, 1] = math.tan(random.uniform(-degree, degree) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-degree, degree) * math.pi / 180)
    print(S)
    new_im = cv2.warpPerspective(im_copy, S, dsize=(w, h), borderValue=(114, 114, 114))
    return new_im


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):

    im_copy = im.copy()
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-10, 10, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        # lut_hue = ((x) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        # lut_sat = np.clip(x, 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        # lut_val = np.clip(x, 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(hue, lut_hue), cv2.LUT(hue, lut_hue)))
        im_copy = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im_copy)
    return im_copy


def translate(im, t=0.1):
    """平移"""
    im_copy = im.copy()
    h,w,_ = im.shape
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - t, 0.5 + t) * w * 0.5
    T[1, 2] = random.uniform(0.5 - t, 0.5 + t) * h * 0.5
    new_t = cv2.warpPerspective(im_copy, T, dsize=(w, h), borderValue=(114, 114, 114))
    return new_t


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image

    f = "./tmp/Cats_Test49.jpg"
    f2 = "./tmp/golf.jpg"

    im = cv2.imread(f)
    im2 = cv2.imread(f2)

    # im_t = translate(im, t=0.01)
    # cv2.imshow('1',im)
    # cv2.imshow('2', im_t)
    # cv2.waitKey(0)

    # flip
    # im_up = np.flipud(im)
    # im_right = np.fliplr(im)
    # cv2.imshow("1", im)
    # cv2.imshow("2", im_up)
    # cv2.imshow("3", im_right)
    # cv2.waitKey(0)

    # hsv
    # cv2.imshow("1",im)
    # im_hsv = hsv(im)
    # cv2.imshow('2', im_hsv)
    # cv2.waitKey(0)

    # mixup
    # im_resize = cv2.resize(im,(320,320))
    # im2_resize = cv2.resize(im2,(320,320))
    # mixup_im, _ = mixup(im_resize, [], im2_resize, [])
    # cv2.imwrite("../data/tmp/mixup.jpg",mixup_im)

    # 错切
    # s_im = shear(im, degree=45)
    # cv2.imshow("origin", im)
    # cv2.imshow("s", s_im)
    # cv2.waitKey(0)

    # 旋转缩放
    # rs_im = rotate_scale(im, degrees=45, scale=0.5)
    # cv2.imshow("origin", im)
    # cv2.imshow("rs", rs_im)
    # cv2.waitKey(0)

    # perspective透视变换
    # p_im = perspective(im, p=0.001)
    # cv2.imshow("oring", im)
    # cv2.imshow("p", p_im)
    # cv2.waitKey(0)

    # 4图拼接
    # mosaic()

    # 分割填补
    from generate_coco_data import CoCoDataGenrator
    from visual_ops import draw_instance
    #
    file = "./instances_val2017.json"
    coco = CoCoDataGenrator(
        coco_annotation_file=file,
        train_img_nums=2,
        include_mask=True,
        include_keypoint=False,
        batch_size=2)
    data = coco.next_batch()
    gt_imgs = data['imgs']
    gt_boxes = data['bboxes']
    gt_classes = data['labels']
    gt_masks = data['masks']
    valid_nums = data['valid_nums']
    im_new, out_boxes, out_masks = copy_paste(
        im_origin=gt_imgs[0],
        boxes_origin=gt_boxes[0][:valid_nums[0]],
        im_masks=gt_imgs[1],
        masks=gt_masks[1][:,:,:valid_nums[1]],
        mask_boxes=gt_boxes[1][:valid_nums[1]])
    final_masks = np.concatenate([gt_masks[0][:,:,:valid_nums[0]], out_masks], axis=-1)
    im_new = draw_instance(im_new, final_masks)

    img0 = gt_imgs[0]
    img0 = draw_instance(img0, gt_masks[0][:, :, :valid_nums[0]])

    img1 = gt_imgs[1]
    img1 = draw_instance(img1, gt_masks[1][:, :, :valid_nums[1]])

    cv2.imshow("origin", img0)
    cv2.imshow("copy", img1)
    cv2.imshow("paste", im_new)
    cv2.waitKey(0)
