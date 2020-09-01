## 运行YOLOv3图像检测样例


### 一：准备环境

请您在环境中安装1.7或以上版本的Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。


### 二：下载模型以及测试数据


1）**获取预测模型**

点击[链接](https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/yolov3_infer.tar.gz)下载模型， 该模型在imagenet数据集训练得到的，如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleDetection)。


2）**获取预测样例图片**

下载[样例图片](https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg)。

图片如下：
<p align="left">
    <br>
<img src='https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg' width = "200" height = "200">
    <br>
<p>


### 三：运行预测

文件`utils.py`包含了图像的预处理等帮助函数。
文件`infer_yolov3.py` 包含了创建predictor，读取示例图片，预测，获取输出的等功能。

运行：
```
python infer_yolov3.py --model_file=./yolov3_infer/__model__ --params_file=./yolov3_infer/__params__ --use_gpu=1
```

输出结果如下所示：

```
category id is 0.0, bbox is [ 98.47467 471.34283 120.73273 578.5184 ]
category id is 0.0, bbox is [ 51.752716 415.51324   73.18762  515.24005 ]
category id is 0.0, bbox is [ 37.176304 343.378     46.64221  380.92963 ]
category id is 0.0, bbox is [155.78638 328.0806  159.5393  339.37192]
category id is 0.0, bbox is [233.86328 339.96912 239.35403 355.3322 ]
category id is 0.0, bbox is [ 16.212902 344.42365   25.193722 377.97137 ]
category id is 0.0, bbox is [ 10.583471 356.67862   14.9261   372.8137  ]
category id is 0.0, bbox is [ 79.76479 364.19492  86.07656 385.64255]
category id is 0.0, bbox is [312.8938  311.9908  314.58527 316.60056]
category id is 33.0, bbox is [266.97925   51.70044  299.45105   99.996414]
category id is 33.0, bbox is [210.45593 229.92128 217.77551 240.97136]
category id is 33.0, bbox is [125.36278 159.80171 135.49306 189.8976 ]
category id is 33.0, bbox is [486.9354  266.164   494.4437  283.84637]
category id is 33.0, bbox is [259.01584 232.23044 270.69266 248.58704]
category id is 33.0, bbox is [135.60567 254.57668 144.96178 276.9275 ]
category id is 33.0, bbox is [341.91315 255.44394 345.0335  262.3398 ]
```

<p align="left">
    <br>
<img src='https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite_res.jpg' width = "200" height = "200">
    <br>
<p>


### 相关链接
- [Paddle Inference使用Quick Start！]()
- [Paddle Inference Python Api使用]()
