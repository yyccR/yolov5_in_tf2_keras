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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    # model_path = "h5模型路径, 默认在 ./logs/yolov5-tf-300.h5"
    model_path = "./logs/yolov5-tf-100.h5"
    # image_path = "提供你要测试的图片路径"
    # image_path = "./data/tmp/traffic_road.jpg"
    image_path = "./data/coco_2017_val_images/289343.jpg"
    # image_path = "./data/tmp/Cats_Test49.jpg"
    image = cv2.imread(image_path)
    # 可以选择 ['5l', '5s', '5m', '5x'], 跟随训练
    yolov5_type = "5s"
    image_shape = (640, 640, 3)
    num_class = 91
    # num_class = 3
    batch_size = 1

    # 这里anchor归一化到[0,1]区间
    anchors = np.array([[10, 13], [16, 30], [33, 23],
                        [30, 61], [62, 45], [59, 119],
                        [116, 90], [156, 198], [373, 326]]) / 640.
    anchors = np.array(anchors, dtype=np.float32)
    # 分别对应1/8, 1/16, 1/32预测输出层
    anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    # data generator
    # coco_data = CoCoDataGenrator(
    #     coco_annotation_file='./data/instances_val2017.json',
    #     train_img_nums=1,
    #     img_shape=image_shape,
    #     batch_size=batch_size,
    #     max_instances=num_class,
    #     include_mask=False,
    #     include_crowd=False,
    #     include_keypoint=False
    # )
    # 类别名, 也可以自己提供一个数组, 不通过coco
    classes = ['none', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'none', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
               'bear', 'zebra', 'giraffe', 'none', 'backpack', 'umbrella', 'none', 'none', 'handbag',
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'none', 'wine glass',
               'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'none',
               'dining table', 'none', 'none', 'toilet', 'none', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'none', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # classes = ['_background_', 'cat', 'dog']

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

    # 预测结果: [nms_nums, (x1, y1, x2, y2, conf, cls)]
    predicts = yolo.predict(image)
    print(predicts)
    if predicts.shape[0]:
        pred_image = image.copy()
        for box_obj_cls in predicts[0]:
            if box_obj_cls[4] > 0.5:
                label = int(box_obj_cls[5])
                class_name = classes[label]
                xmin, ymin, xmax, ymax = box_obj_cls[:4]
                pred_image = draw_bounding_box(pred_image, class_name, box_obj_cls[4], int(xmin), int(ymin),
                                               int(xmax), int(ymax))
        cv2.imwrite("./data/tmp/predicts.jpg", pred_image)
        # cv2.imshow("prediction", pred_image)
        # cv2.waitKey(0)

if __name__ == "__main__":
    main()