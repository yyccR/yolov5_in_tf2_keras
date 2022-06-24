## YOLOv5 in tesnorflow2.x-keras

- [yolov5数据增强jupyter示例](./data/arguments_jupyter.ipynb)
- [Bilibili视频讲解地址: 《yolov5 解读,训练,复现》](https://www.bilibili.com/video/BV1JR4y1g77H/)
- [Bilibili视频讲解PPT文件: yolov5_bilibili_talk_ppt.pdf](./yolov5_bilibili_talk_ppt.pdf)
- [Bilibili视频讲解PPT文件: yolov5_bilibili_talk_ppt.pdf (gitee链接)](https://gitee.com/yyccR/yolov5_in_tf2_keras/blob/master/yolov5_bilibili_talk_ppt.pdf)

### 模型测试

- 训练 [猫狗检测 3.7k](https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection?resource=download)
  
<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_train_loss.png" width="1000" height="500"/> 

- 检测效果

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_train_images.png" width="350" height="230"/> (<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_train_images2.png" width="350" height="230"/>)

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_train_images3.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_train_images4.png" width="350" height="230"/>

- `TFLite` Android 测试效果：

<img src="https://user-images.githubusercontent.com/14156500/175484983-899da03f-5e3f-484a-aec8-383204c58921.png" width="200"/> <img src="https://user-images.githubusercontent.com/14156500/175487023-0c9a2e09-66da-41cc-a21d-7276560c6e0b.png" width="200"/> <img src="https://user-images.githubusercontent.com/14156500/175487105-eca7243b-0f64-427a-9c0c-13ce5866c12e.png" width="200"/> <img src="https://user-images.githubusercontent.com/14156500/175487144-0265775d-9373-43cf-8005-afe87d39845f.png" width="200"/>

- mAP@0.5/mAP@0.5:0.95/精度/召回率

| class | mAP@0.5  | mAP@0.5:0.95 | precision |  recall  |
| :-: | :-: | :-: | :-: | :-: |
|  cat  | 0.962680 |   0.672483   | 0.721003  | 0.958333 |
|  dog  | 0.934285 |   0.546893   | 0.770701  | 0.923664 |
| total | 0.948482 |   0.609688   | 0.745852  | 0.940999 |


### Requirements

```python
pip3 install -r requirements.txt
```

### Get start
0. 下载数据集
```python
https://www.kaggle.com/datasets/andrewmvd/dog-and-cat-detection/download
或者从releasev1.0下载:
https://github.com/yyccR/yolov5_in_tf2_keras/releases/download/v1.0/JPEGImages.zip

解压数据将images目录修改为JPEGImages, 放到 ./data/cat_dog_face_data下
```

1. 训练
```python
python3 train.py
```

2. tensorboard
```python
tensorboard --host 0.0.0.0 --logdir ./logs/ --port 8053 --samples_per_plugin=images=40
```    

3. 查看
```python
http://127.0.0.1:8053
```    

4. 测试, 修改`detect.py`里面`input_image`和`model_path`
```python
python3 detect.py
```

5. 评估验证
```python
python3 val.py
```

6. 导出`TFLite`格式
```python
python3 ./data/h5_to_tfite.py
```

### 训练自己的数据

1. [labelme](https://github.com/wkentaro/labelme)打标自己的数据
2. 打开`data/labelme2coco.py`脚本, 修改如下地方
```angular2html
input_dir = '这里写labelme打标时保存json标记文件的目录'
output_dir = '这里写要转CoCo格式的目录，建议建一个空目录'
labels = "这里是你打标时所有的类别名, txt文本即可, 每行一个类, 类名无需加引号"
```
3. 执行`data/labelme2coco.py`脚本会在`output_dir`生成对应的json文件和图片
4. 修改`train.py`文件中`train_coco_json`, `val_coco_json`, `num_class`, `classes`
5. 开始训练, `python3 train.py`