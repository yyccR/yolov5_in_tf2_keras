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

4. 测试, 修改`detect.py`里面`input_image`和`model_path`
```python
python3 detect.py
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
4. 修改`train.py`文件中`coco_annotation_file`以及`num_class`, 
   注意`classes`通过`CoCoDataGenrator(*).coco.cats[label_id]['name']`可获得，由于coco中类别不连续，所以通过coco.cats拿到的数组下标拿到的类别可能不准.
5. 开始训练, `python3 train.py`