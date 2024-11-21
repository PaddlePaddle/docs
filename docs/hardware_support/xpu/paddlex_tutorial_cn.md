# 昆仑芯 XPU 基于 PaddleX 的使用指南

## 环境准备

### 环境说明

* 本教程介绍如何基于昆仑芯 XPU 进行 ResNet50 的训练，总共需要 4 卡进行训练

* 考虑到环境差异性，我们推荐使用教程提供的标准镜像完成环境准备：

  * 镜像链接： ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310

### 环境安装

1. 安装 PaddlePaddle

*该命令会自动安装飞桨主框架每日自动构建的 nightly-build 版本*

*由于 xpu 代码位于飞桨主框架中，因此我们不需要安装额外的 Custom Device 包*

```shell
python -m pip install paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu/
```

2. 安装 PaddleX 代码库

```shell
git clone https://github.com/PaddlePaddle/PaddleX.git

# 如果速度较慢，可以考虑从 gitee 拉取
# git clone https://gitee.com/paddlepaddle/PaddleX.git

cd PaddleX

# 安装 PaddleX whl
# -e：以可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel
pip install -e .
```

## 基于 PaddleX 训练 ResNet50

### 一、安装 PaddleX 依赖

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 安装 PaddleX 相关依赖，由于我们使用的是图像分类模型，因此安装图像分类库
paddlex --install PaddleClas

# 完成安装后会有如下提示：
# All packages are installed.
```

### 二、数据准备

为了快速上手验证，我们基于 flowers 102 数据集进行快速体验：

1. 下载数据集

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 下载并解压数据
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/cls_flowers_examples.tar -P ./dataset
tar -xf ./dataset/cls_flowers_examples.tar -C ./dataset/
```

2. 数据校验

```shell
# PaddleX 支持对数据集进行校验，确保数据集格式符合 PaddleX 的相关要求。同时在数据校验时，能够对数据集进行分析，统计数据集的基本信息。
python main.py -c paddlex/configs/image_classification/ResNet50.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/cls_flowers_examples

# 命令运行成功后会在 log 中打印出 Check dataset passed ! 信息
```

更多关于 PaddleX 数据集说明的内容，可以查看 [PaddleX 图像分类模块数据准备](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/cv_modules/image_classification.md#41-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)

### 三、模型训练

进入 `PaddleX` 目录下，执行如下命令启动 4 卡 XPU（0 ~ 3 号卡）训练，其中：

* 参数 `-o Global.device` 指定的是即将运行的设备，这里需要传入的是 `xpu:0,1,2,3` ，通过指定该参数，PaddleX 调用飞桨的设备指定接口 `paddle.set_device` 来指定运行设备为 `xpu` ，在进行模型训练时，飞桨将自动调用 xpu 算子用于执行模型计算。关于设备指定的更多细节，可以参考官方 api [paddle.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html#set-device)。

* 参数 `-c paddlex/configs/image_classification/ResNet50.yaml` 表示读取指定目录下的配置文件，配置文件中指定了模型结构，训练超参等所有训练模型需要用到的配置，该文件中指定的模型结构为 `ResNet50`

```shell
python main.py -c paddlex/configs/image_classification/ResNet50.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o Global.output=resnet50_output \
    -o Global.device="xpu:0,1,2,3"
```

上述命令会在 `PaddleX` 目录下产生一个 `resnet50_output/` 目录，该目录会存放训练过程中的模型参数

### 四、模型推理

#### 基于 PaddleInference 推理

训练完成后，最优权重放在 `resnet50_output/best_model/` 目录下，其中 `inference/inference.pdiparams`、`inference/inference.pdiparams.info`、`inference/inference.pdmodel` 3 个文件为静态图文件，用于推理使用，使用如下命令进行推理

```shell
python main.py -c paddlex/configs/image_classification/ResNet50.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./resnet50_output/best_model/inference" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg" \
    -o Global.device="xpu:0"
```

#### 转换 ONNX 模型

如果您有额外的部署需求需要基于 ONNX 实现，我们也提供了专用的工具用于导出 ONNX 模型，参考如下步骤，即可将第一步导出的静态图模型转换为 ONNX 模型：

a. 安装环境

```shell
# 安装 paddle2onnx，该工具支持将 PaddleInference 模型转换为 ONNX 格式
python -m pip install paddle2onnx
```

b. 模型转换

```shell
paddle2onnx --model_dir=./resnet50_output/best_model/inference \
    --model_filename=inference.pdmodel \
    --params_filename=inference.pdiparams \
    --save_file=./resnet50_output/best_model/inference.onnx \
    --enable_onnx_checker=True
```

该命令会在 `resnet50_output/best_model` 目录下生成 `inference.onnx` 文件
