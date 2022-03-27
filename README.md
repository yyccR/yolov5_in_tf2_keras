## YOLOv5 in tesnorflow2.x-keras

- [yolov5数据增强jupyter示例](./data/arguments_jupyter.ipynb)
- [Bilibili视频讲解地址: 《yolov5 解读,训练,复现》](https://www.bilibili.com/video/BV1JR4y1g77H/)
- [Bilibili视频讲解PPT文件: yolov5_bilibili_talk_ppt.pdf](./yolov5_bilibili_talk_ppt.pdf)
- [Bilibili视频讲解PPT文件: yolov5_bilibili_talk_ppt.pdf (gitee链接)](https://gitee.com/yyccR/yolov5_in_tf2_keras/blob/master/yolov5_bilibili_talk_ppt.pdf)

### 模型测试

- 训练 COCO2017(val 5k)
  
<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_train.png" width="1000" height="500"/> 

- 检测效果

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_sample1.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_sample2.png" width="350" height="230"/>

<img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_sample3.png" width="350" height="230"/>  <img src="https://raw.githubusercontent.com/yyccR/Pictures/master/yolov5/yolov5_sample4.png" width="350" height="230"/>

- 精度/召回率

### Requirements

```python
pip3 install -r requirements.txt
```

### Get start

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


### 训练自己的数据

1. [labelme](https://github.com/wkentaro/labelme)打标自己的数据, 导出为coco格式

2. 参考`CoCoDataGenrator`类实现自己的generator

3. `python3 yolov3.py` 训练你的数据