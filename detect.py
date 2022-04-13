import sys

sys.path.append('../yolov5_in_tf2_keras')

import cv2
import os
import numpy as np
import random
import tensorflow as tf
from data.visual_ops import draw_bounding_box
from data.generate_coco_data import CoCoDataGenrator
from yolo import Yolo
from loss import ComputeLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    model_path = "h5模型路径, 默认在根目录下 ./yolov5-tf-300.h5"
    image_path = "提供你要测试的图片路径"
    image = cv2.imread(image_path)
    yolov5_type = "5l"
    image_shape = (640, 640, 3)
    num_class = 91
    batch_size = 1

    # 这里anchor归一化到[0,1]区间
    anchors = np.array([[10, 13], [16, 30], [33, 23],
                        [30, 61], [62, 45], [59, 119],
                        [116, 90], [156, 198], [373, 326]]) / image_shape[0]
    anchors = np.array(anchors, dtype=np.float32)
    # 分别对应1/8, 1/16, 1/32预测输出层
    anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    # data generator
    coco_data = CoCoDataGenrator(
        coco_annotation_file='./data/instances_val2017.json',
        train_img_nums=1,
        img_shape=image_shape,
        batch_size=batch_size,
        max_instances=num_class,
        include_mask=False,
        include_crowd=False,
        include_keypoint=False
    )
    # 类别名, 也可以自己提供一个数组, 不通过coco
    classes = coco_data.coco.cats

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

    # 预测结果: [nms_nums, (x1, y1, x2, y2, conf, cls)]
    predicts = yolo.predict(image)[0]
    pred_image = image.copy()
    for box_obj_cls in predicts:
        if box_obj_cls[4] > 0.5:
            label = int(box_obj_cls[5])
            if classes.get(label):
                class_name = classes[label]['name']
                xmin, ymin, xmax, ymax = box_obj_cls[:4]
                pred_image = draw_bounding_box(pred_image, class_name, box_obj_cls[4], int(xmin), int(ymin),
                                             int(xmax), int(ymax))
    cv2.imwrite("./data/tmp/predicts.jpg", pred_image)

if __name__ == "__main__":
    main()