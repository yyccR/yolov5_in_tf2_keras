
import os
import json
import xmltodict

def xml2dict(file):
    if os.path.isfile(file):
        with open(file, "rb") as f:
            try:
                dict = xmltodict.parse(f.read())
                return dict
            except Exception as e:
                print(e)
                return None
    else:
        print("file:{} not exist".format(file))
        return None

def dict2xml(dict_data, save_file):
    if dict_data:
        try:
            xml_data = xmltodict.unparse(dict_data, pretty=True, full_document=False)
            # print(xml_data)
            if save_file:
                with open(save_file, "w") as f:
                    f.write(xml_data)
            # print(xml_data)
            return xml_data
        except Exception as e:
            print(e)
            return None

def save_voc_xml(im_shape, bound_box, name, save_file_path):
    """
    :param im_shape: (height, width, depth)
    :param bound_box: [[xmin, ymin, xmax, ymax], [...], ...]
    :param name: 0,1234,12345, ...
    :param save_file_path: e.g: /data/tf-faster-rcnn-win10/ckpt/faster_rcnn/data/VOCdevkit2007/VOC2007/Annotations
    :return:
    """
    template_data = {
        "annotation": {
            "folder": "Annotation",
            "filename":"1",
            "path": "1",
            "size":{
                "width":"1",
                "height":"1",
                "depth":"1"
            },
            "segmented":"0",
            "object":[]
        }
    }

    pre_path = os.path.split(save_file_path)[0]
    img_path = os.path.join(pre_path, "JPEGImages_png", str(name)+".jpg")
    template_data["annotation"]['filename'] = str(name)+".jpg"
    template_data['annotation']['path'] = img_path
    template_data["annotation"]['size']['height'] = im_shape[0]
    template_data["annotation"]['size']['width'] = im_shape[1]
    template_data["annotation"]['size']['depth'] = im_shape[2]
    for box in bound_box:
        template_data["annotation"]['object'].append({
            "name": "watermark",
            "pose": "Unspecified",
            "truncated": "0",
            "difficult": "0",
            "bndbox": {
                "xmin": box[0],
                "ymin": box[1],
                "xmax": box[2],
                "ymax": box[3]
            }
        })
    if bound_box:
        xml_save_file = os.path.join(save_file_path, str(name)+".xml")
        dict2xml(template_data, xml_save_file)

if __name__ == "__main__":
    file = "detect_data/Annotations/Cats_Test0.xml"
    xml_dict = xml2dict(file)
    print(xml_dict['annotation']['object'])
    xml_data = xmltodict.unparse(xml_dict, pretty=True, full_document=False)
    print(xml_data)

    # import cv2
    # f = "./xxl2.png"
    # im = cv2.imread(f, cv2.IMREAD_UNCHANGED)[:, 80:1000, :]
    # im_resize = cv2.resize(im, (296, 414))
    # cv2.imwrite('./xxl_s.jpg', im_resize)
    # cv2.imshow('', im_resize)
    # cv2.waitKey(0)