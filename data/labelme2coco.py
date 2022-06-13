#!/usr/bin/env python

import glob
import json
import sys
import uuid
import random
import datetime
import collections
import os.path as osp

import imgviz
import numpy as np

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install \n")
    sys.exit(1)


def labelme2coco(input_dir='', output_dir='', labels=[], train_val_split=0.8):

    train_out_ann_file = osp.join(output_dir, "train_annotations.json")
    val_out_ann_file = osp.join(output_dir, "val_annotations.json")
    label_files = glob.glob(osp.join(input_dir, "*.json"))

    train_dict = {"images": [], "annotations": [], "categories": []}
    val_dict = {"images": [], "annotations": [], "categories": []}

    class_name_to_id = {}
    for i, label in enumerate(labels):
        class_id = i
        class_name = label
        class_name_to_id[class_name] = class_id
        train_dict["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name)
        )
        val_dict["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name)
        )

    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(output_dir, "JPEGImages", base + ".jpg")

        # 保存图片到对应目录
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        if random.random() < train_val_split:
            json_dict = train_dict
        else:
            json_dict = val_dict

        json_dict["images"].append(
            dict(
                file_name=osp.relpath(out_img_file, osp.dirname(train_out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                id=image_id,
            )
        )

        masks = {}
        segments = collections.defaultdict(list)
        for shape in label_file.shapes:

            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")

            if group_id is None:
                group_id = uuid.uuid1()
            instance = (label, group_id)

            if shape_type == "polygon":
                mask = labelme.utils.shape_to_mask(
                    img.shape[:2], points, shape_type
                )

                # 此处是为了保证那些 由于被遮挡而分开成多部分标记的 目标
                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask
                # print(masks[instance].shape)
                segments[instance].append(points)

            elif shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                o_width = abs(x2 - x1)
                o_height = abs(y2 - y1)
                if label not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[label]
                json_dict["annotations"].append(
                    dict(
                        id=len(json_dict["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=[],
                        area=o_width * o_height,
                        bbox=[x1, y1, o_width, o_height],
                        iscrowd=0,
                    )
                )
        segments = dict(segments)
        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            json_dict["annotations"].append(
                dict(
                    id=len(json_dict["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segments[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

    with open(train_out_ann_file, "w") as f:
        json.dump(train_dict, f)
    with open(val_out_ann_file, "w") as f:
        json.dump(val_dict, f)


if __name__ == "__main__":
    labelme2coco(
        input_dir='../../cat_dog_face_data/labels',
        output_dir='../../cat_dog_face_data/coco_labels',
        labels=['head'],
        train_val_split=0.8
    )