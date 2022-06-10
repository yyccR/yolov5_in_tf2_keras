import cv2
import random
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bounding_box(im, cls, scores, x_min, y_min, x_max, y_max, thickness=2, color=(11, 252, 3), txt_size=0.35):
    im_cp = np.array(im.copy(), dtype=np.uint8)
    cv2.rectangle(im_cp, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=thickness)
    txt = "{}:{:.3f}".format(cls, scores)
    x_rect_min = int(x_min)
    y_rect_min = int(y_min - int(38 * txt_size))
    x_rect_max = int(x_min + int(len(txt) * 19 * txt_size))
    y_rect_max = int(y_min)
    x_txt = int(x_min)
    y_txt = int(y_min - 5)
    if y_rect_min < 0:
        y_rect_min = int(y_max)
        y_rect_max = int(y_max + int(38 * txt_size))
        y_txt = int(y_max + 5)
    # draw text box
    cv2.rectangle(im_cp, (x_rect_min, y_rect_min), (x_rect_max, y_rect_max), (11, 252, 3), -1)
    # draw txt
    cv2.putText(im_cp, txt, (x_txt, y_txt), cv2.FONT_HERSHEY_SIMPLEX, txt_size, (0, 0, 0))
    return im_cp


def draw_instance(im, masks, alpha=0.5):
    im_cp = np.array(im.copy(), dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)
    mask_shape = np.shape(masks)

    if len(mask_shape) < 2:
        return im_cp

    elif len(mask_shape) == 2:
        color = _random_colors(15)[np.random.choice(15)]
        for c in range(3):
            im_cp[:, :, c] = im_cp[:, :, c] * (1 - alpha) * masks[:, :] + alpha * masks[:, :] * color[c] * 255 + \
                             im_cp[:, :, c] * (1 - masks[:, :])
    else:
        num_instance = mask_shape[2]
        colors = _random_colors(num_instance)
        for i in range(num_instance):
            color = colors[i]
            for c in range(3):
                im_cp[:, :, c] = im_cp[:, :, c] * (1 - alpha) * masks[:, :, i] + \
                                 alpha * masks[:, :, i] * color[c] * 255 + \
                                 im_cp[:, :, c] * (1 - masks[:, :, i])

    return im_cp


def draw_point(im, x=None, y=None, points=None, color=(255, 0, 0), size=1):
    im_copy = im.copy()
    circle_im = []
    if points is not None:
        for point in points:
            circle_im = cv2.circle(im_copy, (point[0], point[1]), radius=size, color=color, thickness=-1)
    else:
        circle_im = cv2.circle(im_copy, (x, y), radius=size, color=color, thickness=-1)
    return circle_im


def draw_watermark(im, watermark_txt, x, y, alpha=0.95, size=13, font="simhei.ttf", color=(255, 255, 255)):
    """
    :param x: x轴(width)坐标
    :param y: y轴(height)坐标
    :return:
    """
    im_cpy = im.copy()
    cv2img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    pil_font = ImageFont.truetype(font, size, encoding="utf-8")
    draw.text((x, y), watermark_txt, color, font=pil_font)
    im = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    # cv2.putText(im, watermark_txt, (x, y), fontFace=font, fontScale=size, color=color, thickness=thick)
    # cv2.imshow("1", im)
    im_out = cv2.addWeighted(im, alpha, im_cpy, 1 - alpha, gamma=0)
    # cv2.imshow("-1",im_out)

    x_offset, y_offset = draw.textsize(watermark_txt, pil_font)
    x_end = x + x_offset
    y_end = y + y_offset

    output = {
        "im": im_out,
        "box": [x, y, x_end, y_end]
    }
    return output


def _random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


if __name__ == "__main__":
    print(_random_colors(10)[np.random.choice(10)])
    # from data.xml_ops import xml2dict
    # im_file = "detect_data/JPEGImages_png/Cats_Test4.png"
    # xml_file = "detect_data/Annotations/Cats_Test4.xml"
    # im = cv2.imread(im_file)
    # xml = xml2dict(xml_file)
    # xmin = int(xml['annotation']['object']['bndbox']['xmin'])
    # ymin = int(xml['annotation']['object']['bndbox']['ymin'])
    # xmax = int(xml['annotation']['object']['bndbox']['xmax'])
    # ymax = int(xml['annotation']['object']['bndbox']['ymax'])
    #
    # box_im = draw_bounding_box(im, "", "", xmin, ymin, xmax,ymax)
    # cv2.imshow("b", box_im)
    #
    # cv2.waitKey(0)
