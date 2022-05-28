import sys

sys.path.append("../yolov5_in_tf2_keras")

import os
import numpy as np
import cv2
import tensorflow as tf
from yolov5l import Yolov5l
from yolov5x import Yolov5x
from yolov5m import Yolov5m
from yolov5s import Yolov5s
from data.generate_coco_data import CoCoDataGenrator
from data.visual_ops import draw_bounding_box

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class YoloHead(tf.keras.layers.Layer):
    def __init__(self, image_shape, num_class, is_training, strides, anchors, anchors_masks):
        super(YoloHead, self).__init__()
        self.image_shape = image_shape
        self.num_class = num_class
        self.is_training = is_training
        self.strides = strides
        self.anchors = anchors
        self.anchors_masks = anchors_masks
        self.grid = []
        self.anchor_grid = []
        for i, stride in enumerate(strides):
            grid, anchor_grid = self._make_grid(self.image_shape[0] // stride, self.image_shape[1] // stride, i)
            self.grid.append(grid)
            self.anchor_grid.append(anchor_grid)

    def call(self, inputs, *args, **kwargs):
        detect_res = []
        for i, pred in enumerate(inputs):
            if not self.is_training:
                # f_shape = tf.shape(pred)
                # if len(self.grid) < self.anchor_masks.shape[0]:
                #     grid, anchor_grid = self._make_grid(f_shape[1], f_shape[2], i)
                #     self.grid.append(grid)
                #     self.anchor_grid.append(anchor_grid)
                # 这里把输出的值域从[0,1]调整到[0, image_shape]
                pred_xy = (tf.sigmoid(pred[..., 0:2]) * 2. - 0.5 + self.grid[i]) * self.strides[i]
                pred_wh = (tf.sigmoid(pred[..., 2:4]) * 2) ** 2 * self.anchor_grid[i]
                # print(self.grid)
                pred_obj = tf.sigmoid(pred[..., 4:5])
                pred_cls = tf.keras.layers.Softmax()(pred[..., 5:])
                cur_layer_pred_res = tf.keras.layers.Concatenate(axis=-1)([pred_xy, pred_wh, pred_obj, pred_cls])

                # cur_layer_pred_res = tf.reshape(cur_layer_pred_res, [self.batch_size, -1, self.num_class + 5])
                cur_layer_pred_res = tf.keras.layers.Reshape([-1, self.num_class + 5])(cur_layer_pred_res)
                detect_res.append(cur_layer_pred_res)
            else:
                detect_res.append(pred)
        return detect_res if self.is_training else tf.concat(detect_res, axis=1)

    def _make_grid(self, h, w, i):
        cur_layer_anchors = self.anchors[self.anchors_masks[i]] * np.array([[self.image_shape[1], self.image_shape[0]]])
        num_anchors_per_layer = len(cur_layer_anchors)
        yv, xv = tf.meshgrid(tf.range(h), tf.range(w))
        grid = tf.stack((xv, yv), axis=2)
        # 用来计算中心点的grid cell左上角坐标
        grid = tf.tile(tf.reshape(grid, [1, h, w, 1, 2]), [1, 1, 1, num_anchors_per_layer, 1])
        grid = tf.cast(grid, tf.float32)
        # anchor_grid = tf.reshape(cur_layer_anchors * self.strides[i], [1, 1, 1, num_anchors_per_layer, 2])
        anchor_grid = tf.reshape(cur_layer_anchors, [1, 1, 1, num_anchors_per_layer, 2])
        # 用来计算宽高的anchor w/h
        anchor_grid = tf.tile(anchor_grid, [1, h, w, 1, 1])
        anchor_grid = tf.cast(anchor_grid, tf.float32)

        return grid, anchor_grid


class Yolo:

    def __init__(self,
                 num_class,
                 anchors,
                 anchor_masks,
                 image_shape=[640, 640, 3],
                 is_training=True,
                 batch_size=5,
                 net_type='5l',
                 strides=[8, 16, 32],
                 anchors_per_location=3,
                 yolo_max_boxes=100,
                 yolo_iou_threshold=0.3,
                 yolo_conf_threshold=0.5,
                 model_path=None):
        self.image_shape = image_shape
        self.is_training = is_training
        self.batch_size = batch_size
        self.net_type = net_type
        self.strides = strides
        self.anchors_per_location = anchors_per_location
        self.yolo_max_boxes = yolo_max_boxes
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_conf_threshold = yolo_conf_threshold
        self.model_path = model_path

        self.num_class = num_class
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.base_net = None
        if self.net_type == '5l':
            self.base_net = Yolov5l(
                image_shape=self.image_shape,
                batch_size=self.batch_size,
                num_class=self.num_class,
                anchors_per_location=self.anchors_per_location
            ).build_graph()
        elif self.net_type == '5s':
            self.base_net = Yolov5s(
                image_shape=self.image_shape,
                batch_size=self.batch_size,
                num_class=self.num_class,
                anchors_per_location=self.anchors_per_location
            ).build_graph()
        elif self.net_type == '5m':
            self.base_net = Yolov5m(
                image_shape=self.image_shape,
                batch_size=self.batch_size,
                num_class=self.num_class,
                anchors_per_location=self.anchors_per_location
            ).build_graph()
        elif self.net_type == '5x':
            self.base_net = Yolov5x(
                image_shape=self.image_shape,
                batch_size=self.batch_size,
                num_class=self.num_class,
                anchors_per_location=self.anchors_per_location
            ).build_graph()
        else:
            assert self.net_type in ['5l', '5s', '5m', '5x'], "Net type not in {}".format(['5l', '5s', '5m', '5x'])

        self.grid = []
        self.anchor_grid = []
        self.yolov5 = self.build_graph()
        if not is_training:
            assert model_path, "Inference mode need the model_path!"
            assert os.path.isfile(model_path), "Can't find the model weight file!"
            self.yolov5.load_weights(model_path, by_name=True)
            # self.yolov5 = tf.keras.models.load_model(model_path)
            # self.load_weights(model_path, by_name=True)
            print("loading model weight from {}".format(model_path))

    def load_weights(self, model_path, by_name=True, exclude=None):
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(model_path, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            layers = self.yolov5.inner_model.layers if hasattr(self.yolov5, "inner_model") \
                else self.yolov5.layers

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                hdf5_format.load_weights_from_hdf5_group(f, layers)

    def yolo_head(self, features, is_training):
        """ yolo最后输出层
        :param features:
        :return: train mode: [[batch, h, w, num_anchors_per_layer, num_class + 5], [...], [...]]
                 infer mode: [batch, -1, num_class + 5]
        """
        # num_anchors_per_layer = len(self.anchors[0])
        detect_res = []
        for i, pred in enumerate(features):
            if not is_training:
                f_shape = tf.shape(pred)
                if len(self.grid) < self.anchor_masks.shape[0]:
                    grid, anchor_grid = self._make_grid(f_shape[1], f_shape[2], i)
                    self.grid.append(grid)
                    self.anchor_grid.append(anchor_grid)

                # 这里把输出的值域从[0,1]调整到[0, image_shape]
                pred_xy = (tf.sigmoid(pred[..., 0:2]) * 2. - 0.5 + self.grid[i]) * self.strides[i]
                pred_wh = (tf.sigmoid(pred[..., 2:4]) * 2) ** 2 * self.anchor_grid[i]
                # print(self.grid)
                pred_obj = tf.sigmoid(pred[..., 4:5])
                pred_cls = tf.keras.layers.Softmax()(pred[..., 5:])
                cur_layer_pred_res = tf.keras.layers.Concatenate(axis=-1)([pred_xy, pred_wh, pred_obj, pred_cls])

                # cur_layer_pred_res = tf.reshape(cur_layer_pred_res, [self.batch_size, -1, self.num_class + 5])
                cur_layer_pred_res = tf.keras.layers.Reshape([-1, self.num_class + 5])(cur_layer_pred_res)
                detect_res.append(cur_layer_pred_res)
            else:
                detect_res.append(pred)
        return detect_res if is_training else tf.concat(detect_res, axis=1)

    def _make_grid(self, h, w, i):
        cur_layer_anchors = self.anchors[self.anchor_masks[i]] * np.array([[self.image_shape[1], self.image_shape[0]]])
        num_anchors_per_layer = len(cur_layer_anchors)
        yv, xv = tf.meshgrid(tf.range(h), tf.range(w))
        grid = tf.stack((xv, yv), axis=2)
        # 用来计算中心点的grid cell左上角坐标
        grid = tf.tile(tf.reshape(grid, [1, h, w, 1, 2]), [1, 1, 1, num_anchors_per_layer, 1])
        grid = tf.cast(grid, tf.float32)
        # anchor_grid = tf.reshape(cur_layer_anchors * self.strides[i], [1, 1, 1, num_anchors_per_layer, 2])
        anchor_grid = tf.reshape(cur_layer_anchors, [1, 1, 1, num_anchors_per_layer, 2])
        # 用来计算宽高的anchor w/h
        anchor_grid = tf.tile(anchor_grid, [1, h, w, 1, 1])
        anchor_grid = tf.cast(anchor_grid, tf.float32)

        return grid, anchor_grid

    def nms(self, predicts, conf_thres=0.45, iou_thres=0.2, max_det=300, max_nms=3000):
        """ 原yolov5简化版nms, 不用multi label, 不做merge box

        :param predicts:
        :param conf_thres:
        :param iou_thres:
        :param max_det:
        :return: [batch, nms_nums, (x1, y1, x2, y2, conf, cls)]
        """
        output = []

        # 这里遍历每个batch,也就是每张图, 输出的3层预测已经做了合并处理成[batch, -1, 5+num_class]
        for i, predict in enumerate(predicts):
            # predict = predict.numpy()
            # 首先只要那些目标概率大于阈值的
            obj_mask = predict[..., 4] > conf_thres
            predict = predict[obj_mask]

            # 没有满足的数据则跳过去下一张
            if not predict.shape[0]:
                continue

            # 类别概率乘上了目标概率, 作为最终判别概率
            predict[:, 5:] *= predict[:, 4:5]

            x1 = np.maximum(predict[:, 0] - predict[:, 2] / 2, 0)
            y1 = np.maximum(predict[:, 1] - predict[:, 3] / 2, 0)
            x2 = np.minimum(predict[:, 0] + predict[:, 2] / 2, self.image_shape[1])
            y2 = np.minimum(predict[:, 1] + predict[:, 3] / 2, self.image_shape[0])
            box = np.concatenate([x1[:, None], y1[:, None], x2[:, None], y2[:, None]], axis=-1)
            # Detections matrix [n, (x1, y1, x2, y2, conf, cls)]
            max_cls_ids = np.array(predict[:, 5:].argmax(axis=1), dtype=np.float32)
            max_cls_score = predict[:, 5:].max(axis=1)
            print(np.max(max_cls_score), np.min(max_cls_score))
            predict = np.concatenate([box, max_cls_score[:, None], max_cls_ids[:, None]], axis=1)[
                np.reshape(max_cls_score > 0.1, (-1,))]

            n = predict.shape[0]
            if not n:
                continue
            elif n > max_nms:
                predict = predict[predict[:, 4].argsort()[::-1][:max_nms]]

            # 为每个类别乘上一个大数,box再加上这个偏移, 做nms时就可以在类内做
            cls = predict[:, 5:6] * 4096
            # 边框加偏移
            boxes, scores = predict[:, :4] + cls, predict[:, 4]
            nms_ids = tf.image.non_max_suppression(
                boxes=boxes, scores=scores, max_output_size=max_det, iou_threshold=iou_thres)

            output.append(predict[nms_ids.numpy()])

        return np.array(output)

    def build_graph(self):
        # inputs = tf.keras.layers.Input(shape=self.image_shape, batch_size=self.batch_size)
        inputs = tf.keras.layers.Input(shape=self.image_shape)
        yolo_body_outputs = self.base_net(inputs)

        # outputs = self.yolo_head(yolo_body_outputs, is_training=is_training)
        # outputs = self.yolo_head(yolo_body_outputs, is_training=True)
        outputs = YoloHead(
            image_shape=self.image_shape,
            num_class=self.num_class,
            is_training=self.is_training,
            strides=self.strides,
            anchors=self.anchors,
            anchors_masks=self.anchor_masks
        )(yolo_body_outputs)
        # if not self.is_training:
        #     outputs = self.nms(outputs, iou_thres=self.yolo_iou_threshold, conf_thres=self.yolo_conf_threshold)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def predict(self, images):
        """预测
           预测模式下实例化类: is_training=False, weights_path=, batch_size跟随输入建议1, image_shape跟随训练模式,不做调整
        :param images: [batch, h, w, c] or [h, w, c]
        :return [[nms_nums, (x1, y1, x2, y2, conf, cls)], [...], [...], ...]
        """

        if len(np.shape(images)) <= 3:
            images = [images]
            self.batch_size = 1

        final_outputs = []
        for i, im in enumerate(images):
            im_shape = np.shape(im)
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(self.image_shape[0]) / float(im_size_max)

            # resize原始图片
            im_resize = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            im_resize_shape = np.shape(im_resize)
            im_blob = np.zeros(self.image_shape, dtype=np.float32)
            im_blob[0:im_resize_shape[0], 0:im_resize_shape[1], :] = im_resize
            inputs = np.array([im_blob], dtype=np.float32) / 255.

            # 预测, [batch, -1, num_class + 5]
            # outputs = self.yolov5.predict(inputs)
            outputs = self.yolov5.predict(inputs)
            # self.yolov5.load_weights(self.model_path)
            # outputs = self.yolov5(inputs, training=True)
            # outputs = YoloHead(image_shape=self.image_shape,
            #                    num_class=self.num_class,
            #                    is_training=self.is_training,
            #                    strides=self.strides,
            #                    anchors=self.anchors,
            #                    anchors_masks=self.anchor_masks)
            # outputs = self.yolo_head(outputs, is_training=False)
            # 非极大抑制, [nms_nums, (x1, y1, x2, y2, conf, cls)]
            # nms_outputs = self.nms(outputs.numpy(), iou_thres=0.3)[0]
            # print(np.max(outputs[:,:,4]),np.min(outputs[:,:,4]))
            nms_outputs = self.nms(outputs)
            print(nms_outputs.shape)
            # nms_outputs = self.nms(outputs)
            if not nms_outputs.shape[0]:
                continue
            nms_outputs = np.array(nms_outputs[0], dtype=np.float32)

            # resize回原图大小
            boxes = nms_outputs[:, :4]
            b0 = np.maximum(np.minimum(boxes[:, 0] / im_scale, im_shape[1] - 1), 0)
            b1 = np.maximum(np.minimum(boxes[:, 1] / im_scale, im_shape[0] - 1), 0)
            b2 = np.maximum(np.minimum(boxes[:, 2] / im_scale, im_shape[1] - 1), 0)
            b3 = np.maximum(np.minimum(boxes[:, 3] / im_scale, im_shape[0] - 1), 0)
            origin_boxes = np.stack([b0, b1, b2, b3], axis=1)
            nms_outputs[:, :4] = origin_boxes

            final_outputs.append(nms_outputs)
        final_outputs = np.array(final_outputs)

        return final_outputs


if __name__ == "__main__":
    image_shape = (640, 640, 3)
    anchors = np.array([[10, 13], [16, 30], [33, 23],
                        [30, 61], [62, 45], [59, 119],
                        [116, 90], [156, 198], [373, 326]]) / image_shape[0]
    anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int8)
    anchors = np.array(anchors, dtype=np.float32)
    yolo = Yolo(num_class=90, batch_size=1, is_training=True, anchors=anchors, anchor_masks=anchor_masks)
    yolo.yolov5.summary(line_length=200)
    #
    # from tensorflow.python.ops import summary_ops_v2
    # from tensorflow.python.keras.backend import get_graph
    #
    # tb_writer = tf.summary.create_file_writer('./logs')
    # with tb_writer.as_default():
    #     if not yolo3.yolo_model.run_eagerly:
    #         summary_ops_v2.graph(get_graph(), step=0)
