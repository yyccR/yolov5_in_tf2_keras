import math
import numpy as np
import tensorflow as tf


def broadcast_iou(box_1, box_2):
    """ 计算最终iou

    :param box_1:
    :param box_2:
    :return: [batch_size, grid, grid, anchors, num_gt_box]
    """
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """ 计算iou
    :param box1:
    :param box2:
    :param x1y1x2y2:
    :param GIoU:
    :param DIoU:
    :param CIoU:
    :param eps:
    :return:
    """
    # box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # Intersection area
    inter = (tf.minimum(b1_x2, b2_x2) - tf.maximum(b1_x1, b2_x1)) * \
            (tf.minimum(b1_y2, b2_y2) - tf.maximum(b1_y1, b2_y1))

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # 这里计算得到一个最小的边框, 这个边框刚好能将b1,b2包住
        cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1, b2_x1)
        ch = tf.maximum(b1_y2, b2_y2) - tf.minimum(b1_y1, b2_y1)
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                # with torch.no_grad():)
                v = (4 / math.pi ** 2) * tf.pow(tf.atan(w2 / h2) - tf.atan(w1 / h1), 2)
                # with torch.no_grad():
                #     alpha = v / (v - iou + (1 + eps))
                alpha = tf.stop_gradient(v / (v - iou + (1 + eps)))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou


class ComputeLoss:
    """yolov5损失计算"""

    def __init__(self, image_shape, anchors, anchor_masks, num_class,
                 box_loss_gain=0.05, class_loss_gain=0.5, obj_loss_gain=1.0,
                 anchor_ratio_thres=4, only_best_anchor=True, balanced_rate=20,
                 iou_ignore_thres=0.5, layer_balance=[4., 1.0, 0.4]):
        self.image_shape = image_shape
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.num_class = num_class
        self.box_loss_gain = box_loss_gain
        self.class_loss_gain = class_loss_gain
        self.obj_loss_gain = obj_loss_gain
        self.anchor_ratio_thres = anchor_ratio_thres
        self.only_best_anchor = only_best_anchor
        self.balanced_rate = balanced_rate
        self.iou_ignore_thres = iou_ignore_thres
        self.layer_balance = layer_balance

    def _transform_expand_target(self, gt_box_class_anchor, grid_size):
        # y_true: [batch, boxes, (x1, y1, x2, y2, class, best_anchor)]
        batch, num_boxes, _ = np.shape(gt_box_class_anchor)

        # y_true_out: [N, grid, grid, anchors, [x1, y1, x2, y2, obj, class]]
        y_true_out = np.zeros((batch, grid_size, grid_size, len(self.anchor_masks[0]), 6), dtype=np.float32)

        for i in np.arange(batch):
            for j in np.arange(num_boxes):
                # 这里如果是padding的数据则跳过
                if gt_box_class_anchor[i][j][2] == 0:
                    continue
                # 计算中心点
                box = gt_box_class_anchor[i, j, 0:4]
                box_xy = (box[..., 0:2] + box[..., 2:4]) / 2
                anchor_idx = int(gt_box_class_anchor[i, j, 5])

                # 计算目标边框和anchor的宽高比
                w = box[2] - box[0]
                h = box[3] - box[1]
                target_anchor = self.anchors[anchor_idx]
                w_ratio = w / target_anchor[0]
                h_ratio = h / target_anchor[1]
                w_ratio_bool = (w_ratio > 1 / self.anchor_ratio_thres) and \
                               (w_ratio < self.anchor_ratio_thres)
                h_ratio_bool = (h_ratio > 1 / self.anchor_ratio_thres) and \
                               (h_ratio < self.anchor_ratio_thres)

                # 最大iou的anchor同时需要满足比例在[0.25, 4]之间
                if w_ratio_bool and h_ratio_bool:
                    grid_xy = np.array(box_xy // (1 / grid_size), np.int32)

                    # 0,1,2 % 3 = 0,1,2    3,4,5 % 3 = 0,1,2    6,7,8 % 3 = 0,1,2
                    best_anchor_id = anchor_idx % len(self.anchor_masks[0])
                    if self.only_best_anchor:
                        y_true_out[i, grid_xy[0], grid_xy[1], best_anchor_id, :] = \
                            [box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]
                    else:
                        y_true_out[i, grid_xy[0], grid_xy[1], :, :] = \
                            np.array([[box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]])

                    # 扩展更多正样本
                    xy = box_xy * grid_size
                    inv_xy = grid_size - xy
                    if (xy[0] % 1 < 0.5) and (xy[0] > 1.):
                        jxy = xy - np.array([0.5, 0])
                        grid_jxy = np.array(jxy, np.int32)
                        if self.only_best_anchor:
                            y_true_out[i, grid_jxy[0], grid_jxy[1], best_anchor_id, :] = \
                                [box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]
                        else:
                            y_true_out[i, grid_jxy[0], grid_jxy[1], :, :] = \
                                np.array([[box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]])

                    if (xy[1] % 1 < 0.5) and (xy[1] > 1.):
                        kxy = xy - np.array([0, 0.5])
                        grid_kxy = np.array(kxy, np.int32)
                        if self.only_best_anchor:
                            y_true_out[i, grid_kxy[0], grid_kxy[1], best_anchor_id, :] = \
                                [box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]
                        else:
                            y_true_out[i, grid_kxy[0], grid_kxy[1], :, :] = \
                                np.array([[box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]])

                    if (inv_xy[0] % 1 < 0.5) and (inv_xy[0] > 1.):
                        inv_lxy = xy + np.array([0.5, 0])
                        grid_lxy = np.array(inv_lxy, np.int32)
                        if self.only_best_anchor:
                            y_true_out[i, grid_lxy[0], grid_lxy[1], best_anchor_id, :] = \
                                [box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]
                        else:
                            y_true_out[i, grid_lxy[0], grid_lxy[1], :, :] = \
                                np.array([[box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]])

                    if (inv_xy[1] % 1 < 0.5) and (inv_xy[1] > 1.):
                        inv_mxy = xy + np.array([0, 0.5])
                        grid_mxy = np.array(inv_mxy, np.int32)
                        if self.only_best_anchor:
                            y_true_out[i, grid_mxy[0], grid_mxy[1], best_anchor_id, :] = \
                                [box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]
                        else:
                            y_true_out[i, grid_mxy[0], grid_mxy[1], :, :] = \
                                np.array([[box[0], box[1], box[2], box[3], 1, gt_box_class_anchor[i, j, 4]]])
        return y_true_out

    def build_targets(self, predicts, gt_boxes, gt_classes):
        """
        :param predicts: [3, batch, grid, grid, anchors, 5+num_class]
        :param gt_boxes: [batch, num_box, (x1, y1, x2, y2)]
        :param gt_classes: [batch, num_box]
        :return [3, batch, grid, grid, anchors, 6(x1, y1, x2, y2, obj, class)]
        """
        gt_classes = np.expand_dims(gt_classes, axis=-1)

        # [batch, num_box, (w, h)]
        box_wh = gt_boxes[..., 2:4] - gt_boxes[..., 0:2]
        # [batch, num_box, 3, (w, h)]
        box_wh = np.tile(np.expand_dims(box_wh, axis=-2), (1, 1, len(self.anchor_masks[0]), 1))
        # [batch, num_box, 3]
        box_area = box_wh[..., 0] * box_wh[..., 1]

        targets = []
        for i, predict in enumerate(predicts):
            predict = predict.numpy()
            grid_size = predict.shape[1]
            cur_anchor_ids = self.anchor_masks[i]
            cur_anchors = self.anchors[cur_anchor_ids]
            # (3, )
            anchor_area = cur_anchors[..., 0] * cur_anchors[..., 1]

            # 计算iou, 沿用v3的做法, 只要iou最大的anchor
            intersection = np.minimum(box_wh[..., 0], cur_anchors[..., 0]) * \
                           np.minimum(box_wh[..., 1], cur_anchors[..., 1])
            iou = intersection / (box_area + anchor_area - intersection)
            anchor_idx = np.array(cur_anchor_ids[np.argmax(iou, axis=-1)], dtype=np.float32)
            anchor_idx = np.expand_dims(anchor_idx, axis=-1)

            # 拼接最后的结果
            # [batch, num_box, (x1, y1, x2, y2, class, best_anchor_id)]
            gt_box_class_anchor = np.concatenate([gt_boxes, gt_classes, anchor_idx], axis=-1)
            target = self._transform_expand_target(gt_box_class_anchor, grid_size)
            targets.append(target)
        return targets

    def __call__(self, predicts, gt_boxes, gt_classes):
        """
        :param predicts: [3, batch, grid, grid, anchors, 5+num_class]
        :param gt_boxes
        :param gt_classes
        :return
        """
        loss_xy = 0.0
        loss_wh = 0.0
        loss_box = 0.0
        loss_obj = 0.0
        loss_cls = 0.0

        targets = self.build_targets(predicts, gt_boxes, gt_classes)

        for i, predict in enumerate(predicts):
            batch = predict.shape[0]
            grid_size = predict.shape[1]

            # ----------------- 这里处理预测数据 --------------------------
            pred_xy, pred_wh, pred_obj, pred_cls = tf.split(predict, (2, 2, 1, self.num_class), axis=-1)

            # [batch, grid, grid, anchors, 2]
            pred_xy = 2 * tf.sigmoid(pred_xy) - 0.5
            # [batch, grid, grid, anchors, 2]
            pred_wh = (tf.sigmoid(pred_wh) * 2) ** 2 * self.anchors[self.anchor_masks[i]]
            # [batch, grid, grid, anchors, 4]
            pred_xywh = tf.concat((pred_xy, pred_wh), axis=-1)

            # [batch, grid, grid, anchors, 1]
            pred_obj = tf.sigmoid(pred_obj)
            # [batch, grid, grid, anchors, num_class]
            pred_cls = tf.keras.layers.Softmax()(pred_cls)

            grid_y, grid_x = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
            grid = tf.expand_dims(tf.stack([grid_x, grid_y], axis=-1), axis=2)

            # 这里xy从偏移量转成具体中心点坐标, 并且做了归一化, anchors在传进来前也做了归一化
            pred_grid_xy = (pred_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
            pred_x1y1 = pred_grid_xy - pred_wh / 2
            pred_x2y2 = pred_grid_xy + pred_wh / 2

            x1, y1 = tf.split(pred_x1y1, (1, 1), axis=-1)
            x2, y2 = tf.split(pred_x2y2, (1, 1), axis=-1)

            x1 = tf.minimum(tf.maximum(x1, 0.), self.image_shape[1])
            y1 = tf.minimum(tf.maximum(y1, 0.), self.image_shape[0])
            x2 = tf.minimum(tf.maximum(x2, 0.), self.image_shape[1])
            y2 = tf.minimum(tf.maximum(y2, 0.), self.image_shape[0])
            pred_box = tf.concat([x1, y1, x2, y2], axis=-1)

            # ----------------- 这里处理target数据 --------------------------
            # [batch, grid, grid, anchors, 6(x1, y1, x2, y2, obj, class)]
            target = targets[i]
            true_box, true_obj, true_cls = tf.split(target, (4, 1, 1), axis=-1)
            true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
            true_wh = true_box[..., 2:4] - true_box[..., 0:2]

            # 计算true_box的平移缩放量
            # [batch_size, grid, grid, anchors, 2]
            true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)

            # 4. calculate all masks
            # [batch_size, grid, grid, anchors]
            obj_mask = tf.squeeze(true_obj, -1)
            positive_num = tf.cast(tf.reduce_sum(obj_mask), tf.int32) + 1
            negative_num = self.balanced_rate * positive_num

            # print(grid_size)
            # print(tf.where(obj_mask > 0))
            # print(tf.boolean_mask(pred_xy, obj_mask > 0))
            # print(tf.boolean_mask(true_xy, obj_mask > 0))
            # print("--------------")

            # ignore false positive when iou is over threshold
            # [batch_size, grid, grid, anchors, num_gt_box] => [batch_size, grid, grid, anchors, 1]

            best_iou = tf.map_fn(
                lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                    x[1], tf.cast(x[2], tf.bool))), axis=-1),
                (pred_box, true_box, obj_mask),
                tf.float32)
            # [batch_size, grid, grid, anchors, 1]
            ignore_mask = tf.cast(best_iou < self.iou_ignore_thres, tf.float32)
            # 这里做了下样本均衡.
            ignore_num = tf.cast(tf.reduce_sum(ignore_mask), tf.int32)
            if ignore_num > negative_num:
                neg_inds = tf.random.shuffle(tf.where(ignore_mask))[:negative_num]
                neg_inds = tf.expand_dims(neg_inds, axis=1)
                ones = tf.ones(tf.shape(neg_inds)[0], tf.float32)
                ones = tf.expand_dims(ones, axis=1)
                # 更新mask
                ignore_mask = tf.zeros_like(ignore_mask, tf.float32)
                ignore_mask = tf.tensor_scatter_nd_add(ignore_mask, neg_inds, ones)

            # 5. calculate all losses
            # [batch_size, grid, grid, anchors]
            box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
            xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
            # [batch_size, grid, grid, anchors]
            wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

            iou = bbox_iou(tf.boolean_mask(pred_box, obj_mask > 0),
                           tf.boolean_mask(true_box, obj_mask > 0), CIoU=True)
            box_loss = (1. - iou)

            # obj_loss = binary_crossentropy(true_obj, pred_obj)
            conf_focal = tf.pow(obj_mask - tf.squeeze(pred_obj, -1), 2)
            # indices = tf.where(true_obj > 0)
            # true_obj = tf.tensor_scatter_nd_add(true_obj, indices, iou)
            obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
            obj_loss = conf_focal * (obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss)
            # obj_loss =  obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

            # obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
            # 这里除了正样本会计算损失, 负样本低于一定置信的也计算损失
            # obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

            # TODO: use binary_crossentropy instead
            # class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)
            class_loss = obj_mask * tf.keras.losses.sparse_categorical_crossentropy(true_cls, pred_cls)

            # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
            loss_xy += tf.reduce_mean(xy_loss) * batch
            loss_wh += tf.reduce_mean(wh_loss) * batch
            if tf.size(iou) > 0:
                loss_box += tf.reduce_mean(box_loss) * batch * self.box_loss_gain
            loss_obj += tf.reduce_mean(obj_loss) * self.layer_balance[i] * batch * self.obj_loss_gain
            loss_cls += tf.reduce_mean(class_loss) * batch * self.class_loss_gain

        # return xy_loss + wh_loss + obj_loss + class_loss
        return loss_xy, loss_wh, loss_box, loss_obj, loss_cls
        # return loss_xy, loss_wh, loss_xy+loss_wh, loss_obj, loss_cls


if __name__ == "__main__":
    pass
