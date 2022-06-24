import os

import cv2
import tqdm
import numpy as np
import pandas as pd
from yolo import Yolo
import tensorflow as tf
from data.generate_coco_data import CoCoDataGenrator
from layers import nms
from data.visual_ops import draw_bounding_box

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def ap_per_class(correct, pred_conf, pred_cls, target_cls, eps=1e-16):
    """ 计算每个类别的ap
    :param correct: [m, 10], 记录着每个预测值在对应的iou阈值下是否有匹配的真实目标边框
    :param pred_conf: [m]
    :param pred_cls: [m]
    :param target_cls: [n]
    :return:
    """
    # 逆序从大到小
    i = np.argsort(-pred_conf)
    correct, pred_conf, pred_cls = correct[i], pred_conf[i], pred_cls[i]

    # 去重类别
    unique_classes, num_per_classes = np.unique(target_cls, return_counts=True)
    num_classes = unique_classes.shape[0]

    ap = np.zeros((num_classes, correct.shape[1]))
    precision = np.zeros((num_classes))
    recall = np.zeros((num_classes))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        if i.sum() == 0 or num_per_classes[ci] == 0:
            continue

        # 逐步累加, 这里correct里面为true的都是true positive, 即预测的都是真的边框
        fp = (1 - correct[i]).cumsum(0)
        # false的那些预测的都是非真样本, 即false positive
        tp = correct[i].cumsum(0)

        # 当前类别的召回率, tp/当前类别样本数
        r = tp / (num_per_classes[ci] + eps)
        # 拿iou0.5的所有样本召回率
        recall[ci] = r[-1, 0]
        # r[ci] = np.interp(-px, -pred_conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # 当前类别的精度
        p = tp / (tp + fp)
        # 拿iou0.5的所有样本精度
        precision[ci] = p[-1, 0]
        # p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # 每个iou阈值计算AP
        for j in range(tp.shape[1]):
            # 召回率通常是0开始, 精度通常是1开始
            mrec = np.concatenate(([0.0], r[:, j], [1.0]))
            mpre = np.concatenate(([1.0], p[:, j], [0.0]))

            # 对精度从小到大排序再做临近对比
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            x = np.linspace(0, 1, 101)
            # 再通过插值得到具体的精度召回率
            ap[ci, j] = np.trapz(np.interp(x, mrec, mpre), x)
    return unique_classes, ap, precision, recall


def box_iou(box1, box2, eps=1E-7):
    """
    :param box1: [N, 4(x1, y1, x2, y2)]
    :param box2: [M, 4(x1, y1, x2, y2)]
    :return: [N, M]
    """
    box1 = box1[:, None, :]
    xmin = np.maximum(box1[:, :, 0], box2[:, 0])
    ymin = np.maximum(box1[:, :, 1], box2[:, 1])
    xmax = np.minimum(box1[:, :, 2], box2[:, 2])
    ymax = np.minimum(box1[:, :, 3], box2[:, 3])

    w = np.maximum(xmax - xmin, 0)
    h = np.maximum(ymax - ymin, 0)
    inter = w * h
    union = (box1[:, :, 3] - box1[:, :, 1]) * (box1[:, :, 2] - box1[:, :, 0]) + \
            (box2[:, 3] - box2[:, 1]) * (box2[:, 2] - box2[:, 0]) - inter + eps
    return inter / union


def val(model, val_data_generator, classes, desc='val'):
    """ 模型评估

    :param model:
    :param val_data_generator:
    :return:
    """
    mAP50, mAP, final_df = 0., 0., []
    stat = []
    iou_vector = np.linspace(0.5, 0.95, 10)
    progress_bar = tqdm.tqdm(range(val_data_generator.total_batch_size), desc=desc, ncols=100)
    for batch in progress_bar:
        data = val_data_generator.next_batch()
        valid_nums = data['valid_nums']
        gt_imgs = np.array(data['imgs'], dtype=np.float32)
        gt_boxes = np.array(data['bboxes'], dtype=np.float32)
        gt_classes = data['labels']

        # [m, 6(x1,y1,x2,y2,conf,cls_id)]
        if model.is_training:
            # predictions = model.yolov5(gt_imgs / 255., training=True)
            predictions = model.yolov5.predict(gt_imgs / 255.)
            predictions = model.yolo_head(predictions, is_training=False)
            predictions = nms(model.image_shape, predictions.numpy())
        else:
            predictions = model.predict(gt_imgs, image_need_resize=False, resize_to_origin=False)

        for i, prediction in enumerate(predictions):
            if prediction.shape[0]:
                gt_class = gt_classes[i, :valid_nums[i]]
                gt_box = gt_boxes[i, :valid_nums[i], :]

                # [n, m]
                iou = box_iou(gt_box, prediction[:, :4])
                # [n, m]
                correct_label = gt_class[:, None] == prediction[:, 5]
                correct = np.zeros((prediction.shape[0], iou_vector.shape[0]), dtype=np.bool)
                for j, iou_t in enumerate(iou_vector):
                    # 分类正确且iou>阈值
                    x = np.where((iou > iou_t) & correct_label)
                    if x[0].shape[0]:
                        matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                        if x[0].shape[0] > 1:
                            # iou排序
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            # 去重那些 一个预测边框命中多个ground true边框的情况
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            # 去重那些 一个ground true边框匹配上多个预测边框的情况
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True

                stat.append((correct, prediction[:, 4], prediction[:, 5], gt_class))
            # tmp_stat = [np.concatenate(x, axis=0) for x in zip(*stat)]
            # ap = ap_per_class(tmp_stat[0], tmp_stat[1], tmp_stat[2], tmp_stat[3])
            # progress_bar.set_postfix(
            #     ordered_dict={"mAP@0.5:0.95": '{:.5f}'.format(ap.mean()), "mAP@0.5": '{:.5f}'.format(ap[:, 0].mean())})

    # 每个类别计算对应的ap
    if stat:
        stat = [np.concatenate(x, axis=0) for x in zip(*stat)]
        unique_classes, ap, precision, recall = ap_per_class(stat[0], stat[1], stat[2], stat[3])
        # AP@0.5, AP@0.5:0.95
        ap50, ap5095 = ap[:, 0], ap.mean(1)
        mAP50, mAP = ap50.mean(), ap5095.mean()

        df = []
        for ci, cls in enumerate(unique_classes):
            if cls != 'None':
                df.append([classes[int(cls)], ap[ci, 0], ap[ci, :].mean(), precision[ci], recall[ci]])
        df.append(["total", mAP50, mAP, precision.mean(), recall.mean()])
        final_df = pd.DataFrame(data=df, columns=["class", 'mAP@0.5', 'mAP@0.5:0.95', "precision", "recall"])
        print(final_df)
    progress_bar.set_postfix(ordered_dict={"mAP@0.5:0.95": '{:.5f}'.format(mAP), "mAP@0.5": '{:.5f}'.format(mAP50)})
    return mAP50, mAP, final_df


def main():
    # model_path = "h5模型路径, 默认在 ./logs/yolov5-tf-300.h5"
    model_path = "./logs/yolov5s-best.h5"
    # image_path = "提供你要测试的图片路径"
    # image_path = "./data/tmp/Cats_Test49.jpg"
    # image = cv2.imread(image_path)
    val_dataset = './data/cat_dog_face_data/val_annotations.json'
    # 可以选择 ['5l', '5s', '5m', '5x'], 跟随训练
    yolov5_type = "5s"
    # image_shape = (640, 640, 3)
    image_shape = (320, 320, 3)
    # num_class = 91
    num_class = 2
    batch_size = 1

    # 这里anchor归一化到[0,1]区间
    anchors = np.array([[10, 13], [16, 30], [33, 23],
                        [30, 61], [62, 45], [59, 119],
                        [116, 90], [156, 198], [373, 326]]) / 640.
    anchors = np.array(anchors, dtype=np.float32)
    # 分别对应1/8, 1/16, 1/32预测输出层
    anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    # data generator
    val_coco_data = CoCoDataGenrator(
        coco_annotation_file=val_dataset,
        train_img_nums=-1,
        img_shape=image_shape,
        batch_size=batch_size,
        max_instances=num_class,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False
    )
    # 类别名, 也可以自己提供一个数组, 不通过coco
    # classes = ['none', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'none', 'stop sign',
    #            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    #            'bear', 'zebra', 'giraffe', 'none', 'backpack', 'umbrella', 'none', 'none', 'handbag',
    #            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    #            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'none', 'wine glass',
    #            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    #            'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'none',
    #            'dining table', 'none', 'none', 'toilet', 'none', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    #            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'none', 'book', 'clock',
    #            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    classes = ['cat', 'dog']

    yolo = Yolo(
        model_path=model_path,
        image_shape=image_shape,
        batch_size=batch_size,
        num_class=num_class,
        is_training=False,
        anchors=anchors,
        anchor_masks=anchor_masks,
        net_type=yolov5_type
    )
    yolo.yolov5.summary(line_length=100)

    mAP50, mAP, metrics = val(model=yolo, val_data_generator=val_coco_data, classes=classes)


if __name__ == "__main__":
    # x = np.array([[0,0,1,1],[1,1,2,2],[2,2,3,3]])
    # y = np.array([[0.5,0.5,1,1],[1.5,1.5,2.5,2.5]])
    # iou = box_iou(x,y)
    # print(iou)
    main()
