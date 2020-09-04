## 运行Resnet50图像分类样例


### 一：准备环境

请您在环境中安装1.7或以上版本的Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。


### 二：下载模型以及测试数据


1）**获取预测模型**

下载[模型](https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ResNet50.tar.gz)，模型为imagenet 数据集训练得到的，如果你想获取更多的模型训练信息，请访问[这里](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification)。


2）**获取预测样例图片**

下载[样例图片](https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg)。

图片如下：
<p align="left">
    <br>
<img src='https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg' width = "200" height = "200">
    <br>
<p>


### 三：运行预测

文件`img_preprocess.py`包含了图像的预处理。  
文件`infer_resnet.py` 包含了创建predictor，读取示例图片，预测，获取输出的等功能。

运行：
```
python infer_resnet.py --model_file=./ResNet50/model --params_file=./ResNet50/params --use_gpu=1
```

运行的结果为： ('class index: ', 13)。
13表示图片的类别。我们通过imagenet [类别映射表](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)， 可以找到对应的类别，即junco, snowbird，由此说明我们的分类器分类正确。

### 相关链接
- [Paddle Inference使用Quick Start！]()
- [Paddle Inference Python Api使用]()
