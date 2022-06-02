
import os
import cv2
import json
import random
from pathlib import Path
from data.xml_ops import xml2dict
import glob


def voc2coco(root_path):
    voc_labels_dir_path = os.path.join(root_path, "Annotations")
    voc_images_dir_path = os.path.join(root_path, "JPEGImages")
    classes = ["cat", "dog"]

    output_train_annotation = os.path.join(root_path, "train_annotations.json")
    output_val_annotation = os.path.join(root_path, "val_annotations.json")

    train_percent = 0.8

    voc_label_files = glob.glob(os.path.join(Path(voc_labels_dir_path), "*.*"), recursive=True)
    classes_count = dict((c, 0) for c in classes)

    train_dict = {"images": [], "annotations": [], "categories": []}
    val_dict = {"images": [], "annotations": [], "categories": []}
    image_id = 0
    bnd_id = 0
    for label_file in voc_label_files:
        image_id += 1

        # 初始化几个基本属性
        if random.random() < train_percent:
            json_dict = train_dict
        else:
            json_dict = val_dict

        file_name, extension = os.path.basename(label_file).split(".")
        if extension == "xml":
            image_file = os.path.join(voc_images_dir_path, file_name + ".jpg")
            # 图片存在
            if os.path.isfile(image_file):
                image = cv2.imread(image_file)
                height, width, _ = image.shape

                # 添加image属性
                image = {
                    "file_name": os.path.join("JPEGImages", file_name + ".jpg"),
                    "height": height,
                    "width": width,
                    "id": image_id
                }
                print(image_id, "total:", len(voc_label_files) - 1, image)
                json_dict['images'].append(image)

                # 读xml文件
                label_dict = xml2dict(label_file)

                # 存在多个目标边框
                if type(label_dict['annotation']['object']) == list:
                    for objs in label_dict['annotation']['object']:
                        cls = objs['name']
                        classes_count[cls] += 1
                        cls_idx = classes.index(cls)
                        box = objs['bndbox']
                        xmin = int(box['xmin'])
                        ymin = int(box['ymin'])
                        xmax = int(box['xmax'])
                        ymax = int(box['ymax'])
                        o_width = abs(xmax - xmin)
                        o_height = abs(ymax - ymin)

                        # 添加ann属性
                        ann = {
                            "area": o_width * o_height,
                            "iscrowd": 0,
                            "image_id": image_id,
                            "bbox": [xmin, ymin, o_width, o_height],
                            "category_id": cls_idx,
                            "id": bnd_id,
                            "ignore": 0,
                            "segmentation": [],
                        }
                        json_dict["annotations"].append(ann)
                        bnd_id = bnd_id + 1

                else:
                    cls = label_dict['annotation']['object']['name']
                    cls_idx = classes.index(cls)
                    classes_count[cls] += 1
                    box = label_dict['annotation']['object']['bndbox']
                    xmin = int(box['xmin'])
                    ymin = int(box['ymin'])
                    xmax = int(box['xmax'])
                    ymax = int(box['ymax'])
                    o_width = abs(xmax - xmin)
                    o_height = abs(ymax - ymin)
                    # 添加ann属性
                    ann = {
                        "area": o_width * o_height,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": [xmin, ymin, o_width, o_height],
                        "category_id": cls_idx,
                        "id": bnd_id,
                        "ignore": 0,
                        "segmentation": [],
                    }
                    json_dict["annotations"].append(ann)
                    bnd_id = bnd_id + 1

            for cid, cate in enumerate(classes):
                cat = {"supercategory": "none", "id": cid, "name": cate}
                json_dict["categories"].append(cat)

    train_json_fp = open(output_train_annotation, "w")
    val_json_fp = open(output_val_annotation, "w")
    train_json_str = json.dumps(train_dict)
    val_json_str = json.dumps(val_dict)
    train_json_fp.write(train_json_str)
    val_json_fp.write(val_json_str)
    train_json_fp.close()
    val_json_fp.close()

    # 统计各自的比例
    print("最终各类别数量: ", classes_count)
    print("训练图片数量: ", len(train_dict['images']))
    print("测试图片数量: ", len(val_dict['images']))


if __name__ == "__main__":
    voc2coco("./cat_dog_face")
