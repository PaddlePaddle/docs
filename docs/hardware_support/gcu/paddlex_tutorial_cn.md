# 燧原 GCU 基于 PaddleX 的使用指南

## 环境准备

### 环境说明

* 本教程介绍如何基于燧原 S60 GCU 进行 ResNet50 / PP-YOLOE+ / PP-OCRv4 等不同领域模型的评估和推理，总共需要 1 张卡

* 考虑到环境差异性，我们推荐使用教程提供的标准镜像完成环境准备：

  * 镜像链接：ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84

  * 镜像中已经默认安装了燧原软件栈 TopsRider-3.2.109

* 燧原软件栈驱动版本为 1.2.0.301

### 环境安装

1. 安装 PaddlePaddle

*该命令会自动安装飞桨主框架每日自动构建的 nightly-build 版本*

```shell
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```

2. 安装 CustomDevice

*该命令会自动安装飞桨 Custom Device 每日自动构建的 nightly-build 版本*

```shell
python -m pip install paddle-custom-gcu -i https://www.paddlepaddle.org.cn/packages/nightly/gcu/
```

3. 安装 PaddleX 代码库

```shell
git clone https://github.com/PaddlePaddle/PaddleX.git

# 如果速度较慢，可以考虑从 gitee 拉取
# git clone https://gitee.com/paddlepaddle/PaddleX.git

cd PaddleX

# 安装 PaddleX whl
# -e：以可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel
pip install -e .
```

## 基于 PaddleX 进行 ResNet50 推理

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

更多关于 PaddleX 数据集说明的内容，可以查看 [PaddleX 图像分类模块数据准备](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/module_usage/tutorials/cv_modules/image_classification.md#41-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)

3. 下载预训练权重

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 下载预训练权重
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ResNet50_pretrained.pdparams -P ./ResNet50
```

### 三、模型评估

进入 `PaddleX` 目录下，执行如下命令启动评估，其中：

* 参数 `-o Global.device` 指定的是即将运行的设备，这里需要传入的是 `gcu:0` ，通过指定该参数，PaddleX 调用飞桨的设备指定接口 `paddle.set_device` 来指定运行设备为 `gcu` ，在进行模型训练、评估时，飞桨将自动调用 `gcu` 算子用于执行模型计算。关于设备指定的更多细节，可以参考官方 `api` [paddle.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html#set-device)。

* 参数 `-c paddlex/configs/image_classification/ResNet50.yaml` 表示读取指定目录下的配置文件，配置文件中指定了模型结构，训练、评估超参等所有模型需要用到的配置，该文件中指定的模型结构为 `ResNet50`

* 参数 `-o Evaluate.weight_path` 表示读取指定目录下的预训练权重文件

```shell
python main.py -c paddlex/configs/image_classification/ResNet50.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/cls_flowers_examples \
    -o Global.output=ResNet50/output \
    -o Global.device="gcu:0" \
    -o Evaluate.weight_path=./ResNet50/ResNet50_pretrained.pdparams
```

上述命令会在 `PaddleX/ResNet50/` 目录下产生一个 `output/` 目录，该目录会存放评估结果

### 四、模型推理

#### 基于 PaddleInference 推理

下载推理模型和权重

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 下载静态图文件和权重
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/ResNet50_infer.tar -P ./ResNet50
tar -xf ./ResNet50/ResNet50_infer.tar -C ./ResNet50/
```

其中 `ResNet50_infer/inference.pdiparams`、`ResNet50_infer/inference.pdiparams.info`、`ResNet50_infer/inference.pdmodel` 3 个文件为静态图文件，用于推理使用，使用如下命令进行推理

```shell
python main.py -c paddlex/configs/image_classification/ResNet50.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./ResNet50/ResNet50_infer" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg" \
    -o Global.device="gcu:0"
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
paddle2onnx --model_dir=./ResNet50/ResNet50_infer \
    --model_filename=inference.pdmodel \
    --params_filename=inference.pdiparams \
    --save_file=./ResNet50/ResNet50_infer/inference.onnx \
    --enable_onnx_checker=True
```

该命令会在 `ResNet50/ResNet50_infer` 目录下生成 `inference.onnx` 文件

## 基于 PaddleX 进行 PP-YOLOE+ 推理

### 一、安装 PaddleX 依赖

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 安装 PaddleX 相关依赖，由于我们使用的是目标检测模型，因此安装目标检测库
paddlex --install PaddleDetection

# 完成安装后会有如下提示：
# All packages are installed.
```

### 二、数据准备

为了快速上手验证，我们基于 PaddleX 准备的 Demo 数据集进行快速体验：

1. 下载数据集

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 下载并解压数据
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/det_coco_examples.tar -P ./dataset
tar -xf ./dataset/det_coco_examples.tar -C ./dataset/
```

2. 数据校验

```shell
# PaddleX 支持对数据集进行校验，确保数据集格式符合 PaddleX 的相关要求。同时在数据校验时，能够对数据集进行分析，统计数据集的基本信息。
python main.py -c paddlex/configs/object_detection/PP-YOLOE_plus-S.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/det_coco_examples

# 命令运行成功后会在 log 中打印出 Check dataset passed ! 信息
```

更多关于 PaddleX 数据集说明的内容，可以查看 [PaddleX 目标检测模块数据准备](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/module_usage/tutorials/cv_modules/object_detection.md#41-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)

3. 下载预训练权重

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 下载预训练权重
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-YOLOE_plus-S_pretrained.pdparams -P ./PP-YOLOE_plus-S
```

### 三、模型评估

进入 `PaddleX` 目录下，执行如下命令启动评估，其中：

* 参数 `-o Global.device` 指定的是即将运行的设备，这里需要传入的是 `gcu:0` ，通过指定该参数，PaddleX 调用飞桨的设备指定接口 `paddle.set_device` 来指定运行设备为 `gcu` ，在进行模型训练、评估时，飞桨将自动调用 `gcu` 算子用于执行模型计算。关于设备指定的更多细节，可以参考官方 `api` [paddle.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html#set-device)。

* 参数 `-c paddlex/configs/object_detection/PP-YOLOE_plus-S.yaml` 表示读取指定目录下的配置文件，配置文件中指定了模型结构，训练、评估超参等所有模型需要用到的配置，该文件中指定的模型结构为 `PP-YOLOE_plus-S`

* 参数 `-o Evaluate.weight_path` 表示读取指定目录下的预训练权重文件

```shell
python main.py -c paddlex/configs/object_detection/PP-YOLOE_plus-S.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/det_coco_examples \
    -o Global.output=PP-YOLOE_plus-S/output \
    -o Global.device="gcu:0" \
    -o Evaluate.weight_path=./PP-YOLOE_plus-S/PP-YOLOE_plus-S_pretrained.pdparams
```

上述命令会在 `PaddleX/PP-YOLOE_plus-S/` 目录下产生一个 `output/` 目录，该目录会存放评估结果

### 四、模型推理

#### 基于 PaddleInference 推理

下载推理模型和权重

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 下载静态图文件和权重
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-YOLOE_plus-S_infer.tar -P ./PP-YOLOE_plus-S
tar -xf ./PP-YOLOE_plus-S/PP-YOLOE_plus-S_infer.tar -C ./PP-YOLOE_plus-S/
```

其中 `PP-YOLOE_plus-S_infer/inference.pdiparams`、`PP-YOLOE_plus-S_infer/inference.pdiparams.info`、`PP-YOLOE_plus-S_infer/inference.pdmodel` 3 个文件为静态图文件，用于推理使用，使用如下命令进行推理

```shell
python main.py -c paddlex/configs/object_detection/PP-YOLOE_plus-S.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./PP-YOLOE_plus-S/PP-YOLOE_plus-S_infer/" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png" \
    -o Global.device="gcu:0"
```

## 基于 PaddleX 进行 PP-OCR 推理

### 一、安装 PaddleX 依赖

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 安装 PaddleX 相关依赖，由于我们使用的是文本检测/识别模型，因此安装文本检测/识别库
paddlex --install PaddleOCR

# 完成安装后会有如下提示：
# All packages are installed.
```

### 二、数据准备

为了快速上手验证，我们基于 PaddleX 准备的 Demo 数据集进行快速体验，以文本识别为例：

1. 下载数据集

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 下载并解压数据
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar -P ./dataset
tar -xf ./dataset/ocr_rec_dataset_examples.tar -C ./dataset/
```

2. 数据校验

```shell
# PaddleX 支持对数据集进行校验，确保数据集格式符合 PaddleX 的相关要求。同时在数据校验时，能够对数据集进行分析，统计数据集的基本信息。
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples

# 命令运行成功后会在 log 中打印出 Check dataset passed ! 信息
```

更多关于 PaddleX 数据集说明的内容，可以查看 [PaddleX 文本识别模块数据准备](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/module_usage/tutorials/ocr_modules/text_recognition.md#41-%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)

3. 下载预训练权重

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 下载预训练权重
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams -P ./PP-OCRv4_mobile_rec
```

### 三、模型评估

进入 `PaddleX` 目录下，执行如下命令启动评估，其中：

* 参数 `-o Global.device` 指定的是即将运行的设备，这里需要传入的是 `gcu:0` ，通过指定该参数，PaddleX 调用飞桨的设备指定接口 `paddle.set_device` 来指定运行设备为 `gcu` ，在进行模型训练、评估时，飞桨将自动调用 `gcu` 算子用于执行模型计算。关于设备指定的更多细节，可以参考官方 `api` [paddle.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html#set-device)。

* 参数 `-c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml` 表示读取指定目录下的配置文件，配置文件中指定了模型结构，训练、评估超参等所有模型需要用到的配置，该文件中指定的模型结构为 `PP-OCRv4_mobile_rec`

* 参数 `-o Evaluate.weight_path` 表示读取指定目录下的预训练权重文件

```shell
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=evaluate \
    -o Global.dataset_dir=./dataset/ocr_rec_dataset_examples \
    -o Global.output=PP-OCRv4_mobile_rec/output \
    -o Global.device="gcu:0" \
    -o Evaluate.weight_path=./PP-OCRv4_mobile_rec/PP-OCRv4_mobile_rec_pretrained.pdparams
```

上述命令会在 `PaddleX/PP-OCRv4_mobile_rec/` 目录下产生一个 `output/` 目录，该目录会存放评估结果

### 四、模型推理

#### 基于 PaddleInference 推理

下载推理模型和权重

```shell
# 跳转到 PaddleX 根目录下
cd /path/to/paddlex

# 下载静态图文件和权重
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0b2/PP-OCRv4_mobile_rec_infer.tar -P ./PP-OCRv4_mobile_rec
tar -xf ./PP-OCRv4_mobile_rec/PP-OCRv4_mobile_rec_infer.tar -C ./PP-OCRv4_mobile_rec/
```

其中 `PP-OCRv4_mobile_rec_infer/inference.pdiparams`、`PP-OCRv4_mobile_rec_infer/inference.pdiparams.info`、`PP-OCRv4_mobile_rec_infer/inference.pdmodel` 3 个文件为静态图文件，用于推理使用，使用如下命令进行推理

```shell
python main.py -c paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./PP-OCRv4_mobile_rec/PP-OCRv4_mobile_rec_infer/" \
    -o Predict.input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png" \
    -o Global.device="gcu:0"
```
