
import numpy as np
import matplotlib.pyplot as plt
from data.generate_coco_data import CoCoDataGenrator


def save_coco_all_wh(anno_file, image_shape, output_anchor_file):
    """ 生成coco标注数据目标边框宽高
    :param anno_file:
    :param image_shape:
    :param output_anchor_file:
    :return:
    """
    coco = CoCoDataGenrator(coco_annotation_file=anno_file, img_shape=image_shape, batch_size=1, train_img_nums=-1)
    all_wh = []
    for batch in range(coco.total_batch_size):
        data = coco.next_batch()
        # [1, nums, (x1, y1, x2, y2)]
        gt_boxes = data['bboxes']
        # [nums, (w, h)]
        wh = gt_boxes[0][:, 2:4] - gt_boxes[0][:, 0:2]
        wh = list(filter(lambda x: x[0] != 0 and x[1] != 0, wh))
        print("current batch {}, box_wh {}".format(batch, wh))
        all_wh.extend(wh)
    all_wh = np.array(all_wh, dtype=np.int16)
    np.savetxt(output_anchor_file, X=all_wh, fmt="%d", delimiter=",", newline="\n")


def get_wh(wh_file):
    """ 从文件里读取所有的宽高数据
    :param anchors_file:
    :return:
    """
    return np.genfromtxt(fname=wh_file, dtype=np.int32, delimiter=",")


def compute_iou(x, centers):
    """ 计算所有x与对应centers数据的交并比
    :param x:
    :param centers:
    :return:
    """
    # 1. x,centers 复制
    box1 = np.reshape(np.tile(np.expand_dims(x, axis=1), [1, 1, np.shape(centers)[0]]), [-1, 2])
    xw, xh = box1[:, 0], box1[:, 1]
    box2 = np.tile(centers, [np.shape(x)[0], 1])
    cw, ch = box2[:, 0], box2[:, 1]
    # 2. 计算交叉面积
    min_w = np.minimum(xw, cw)
    min_h = np.minimum(xh, ch)
    intersection = min_w * min_h
    # 3. 计算交并比
    x_area = xw * xh
    c_area = cw * ch
    union = x_area + c_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    iou = np.reshape(iou, [-1, np.shape(centers)[0]])
    return iou


def kmeans(x, K, save_cluster_fig=""):
    """ 根据一系列宽高数据聚类生成对应K个中心anchor
    :param x:
    :param K:
    :param save_cluster_fig: 输出聚类效果图保存路径, 为空不输出
    :return: [(w, h)¹, (w, h)², ... (w, h)ᵏ]
    """

    # 随机选K个中心点作为初始值
    centers = x[np.random.choice(a=x.shape[0], size=K, replace=False)]
    pre_max_center_ids = np.zeros(np.shape(x)[0], dtype=np.int64)

    step = 0
    while True:
        iou = compute_iou(x=x, centers=centers)
        max_center_ids = np.argmax(a=iou, axis=1)
        print("step {}, centers: \n {}".format(step, centers))

        if np.all(max_center_ids == pre_max_center_ids):
            if save_cluster_fig:
                for i in range(K):
                    plt.scatter(x[max_center_ids == i, 0], x[max_center_ids == i, 1], label=i, s=10)
                plt.scatter(centers[:, 0], centers[:, 1], s=30, color='k')
                plt.legend()
                plt.savefig(save_cluster_fig)
                plt.show()

            # 根据面积大小重新排序输出centers
            sort_centers = sorted(centers, key=lambda v:v[0]*v[1], reverse=False)
            sort_centers = np.array(sort_centers, dtype=np.int64)
            print("final centers \n {}".format(sort_centers))
            return sort_centers

        centers = np.zeros_like(centers, dtype=np.float32)
        for j in range(K):
            target_x_index = max_center_ids == j
            target_x = x[target_x_index,]
            # print(np.sum(target_x, axis=0), np.sum(target_x_index))
            centers[j] = np.sum(target_x, axis=0) / np.sum(target_x_index)

        pre_max_center_ids = max_center_ids.copy()
        step += 1


if __name__ == "__main__":
    # save_coco_all_wh(
    #     anno_file="./data/instances_val2017.json",
    #     image_shape=[320,320,3],
    #     output_anchor_file='./data/coco2017_320x320_val_all_wh.txt'
    # )

    data = get_wh('./coco2017_320x320_val_all_wh.txt')
    kmeans(data, 9, save_cluster_fig="./anchors_320x320_test.png")
