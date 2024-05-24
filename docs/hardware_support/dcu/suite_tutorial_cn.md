# 环境准备

## 环境说明

* 本教程介绍如何基于海光 DCU 进行 ResNet50 的训练，总共需要 4 卡进行训练

* 考虑到环境差异性，我们推荐使用教程提供的标准镜像完成环境准备：

  * 镜像链接： registry.baidubce.com/device/paddle-dcu:dtk23.10.1-kylinv10-gcc73-py310

## 环境安装

安装 PaddlePaddle

*该命令会自动安装飞桨主框架每日自动构建的 nightly-build 版本*

```shell
python -m pip install --pre paddlepaddle-rocm -i https://www.paddlepaddle.org.cn/packages/nightly/dcu/
```

# 基于 PaddleClas 训练 ResNet50

## 一、安装 PaddleClas 代码库

```shell
git clone https://github.com/PaddlePaddle/PaddleClas.git -b release/2.5.1
cd PaddleClas
python -m pip install -r requirements.txt
python -m pip install .
```

## 二、数据准备

请根据 [数据说明文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.0/docs/zh_CN/tutorials/data.md) 准备 ImageNet1k 数据集，准备完成后解压到 PaddleClas/dataset/目录下，目录结构如下：

```
PaddleClas/dataset/ILSVRC2012/
|_ train/
|  |_ n01440764
|  |  |_ n01440764_10026.JPEG
|  |  |_ ...
|  |_ ...
|  |
|  |_ n15075141
|     |_ ...
|     |_ n15075141_9993.JPEG
|_ val/
|  |_ ILSVRC2012_val_00000001.JPEG
|  |_ ...
|  |_ ILSVRC2012_val_00050000.JPEG
|_ train_list.txt
|_ val_list.txt
```

## 三、模型训练

进入 PaddleClas 目录下，执行如下命令启动 4 卡 DCU（0 ~ 3 号卡）训练，其中：

* 参数 `-c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml` 表示读取指定目录下的配置文件，配置文件中指定了模型结构，训练超参等所有训练模型需要用到的配置，该文件中指定的模型结构为 `ResNet50`

```shell
python -u -m paddle.distributed.launch --devices 0,1,2,3 tools/train.py \
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
    -o Global.output_dir="output/ResNet50" \
    -o Global.device="gpu"
```

上述命令会在 PaddleClas 目录下产生一个 output/ResNet50 目录，该目录会存放训练过程中的模型参数

## 四、模型导出 & 推理

### 模型导出

训练完成后，最后一个 epoch 的权重放在 output/ResNet50/ 目录下的 epoch_120.pdparams 文件中，执行以下命令将模型转成 Paddle 静态图格式存储，以获得更好的推理性能：

* export_model.py 执行的是 `动转静` 操作，飞桨框架会对代码进行分析，将动态图代码（灵活易用）转为 静态图模型（高效），以达到更加高效的推理性能

* 该操作会在指定./deploy/models/ResNet50 下生成 inference.pdiparams、inference.pdiparams.info、inference.pdmodel 3 个文件

```shell
python tools/export_model.py \
    -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
    -o Global.pretrained_model=./output/ResNet50/epoch_120 \
    -o Global.save_inference_dir=./deploy/models/ResNet50 \
    -o Global.device=gpu
```

### 基于 PaddleInference 推理

推理代码位于 PaddleClas/deploy 目录下，执行下列命令进行 DCU 推理：

* 该脚本将会加载上一步保存的静态图，使用飞桨预测库 PaddleInference 进行推理

* PaddleInference 内置了大量的高性能 Kernel，并且可以基于计算图分析，完成细粒度 OP 横向纵向融合，实现了高性能推理

```shell
cd deploy
export FLAGS_conv_workspace_size_limit=2000
python python/predict_cls.py \
    -c ./configs/inference_cls.yaml \
    -o Global.inference_model_dir=./models/ResNet50 \
    -o Global.use_gpu=True \
    -o Global.infer_imgs=./images/ImageNet
```

### 转换 ONNX 模型

如果您有额外的部署需求需要基于 ONNX 实现，我们也提供了专用的工具用于导出 ONNX 模型，参考如下步骤，即可将第一步导出的静态图模型转换为 ONNX 模型：

a. 安装环境

```shell
# 安装 paddle2onnx，该工具支持将 PaddleInference 模型转换为 ONNX 格式
python -m pip install paddle2onnx
```

b. 模型转换

```shell
paddle2onnx --model_dir=./deploy/models/ResNet50/ \
    --model_filename=inference.pdmodel \
    --params_filename=inference.pdiparams \
    --save_file=./deploy/models/ResNet50_onnx/inference.onnx \
    --opset_version=10 \
    --enable_onnx_checker=True
```

该命令会在 deploy/models/ResNet50_onnx 目录下生成 inference.onnx 文件

# 基于 PaddleSeg 训练 DeepLabv3+

## 一、安装 PaddleSeg 代码库

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg -b release/2.9
cd PaddleSeg
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 二、数据准备

请根据 [数据说明文档](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9/docs/data/pre_data_cn.md#cityscapes%E6%95%B0%E6%8D%AE%E9%9B%86) 准备 Cityscapes 数据集，准备完成后解压到 PaddleSeg/data/目录下，目录结构如下：

```shell
PaddleSeg/data/cityscapes
├── leftImg8bit
│   ├── train
│   ├── val
├── gtFine
│   ├── train
│   ├── val
```

## 三、模型训练

进入 PaddleSeg 目录下，执行如下命令启动 4 卡 DCU（0 ~ 3 号卡）训练，其中：

* 参数 `--device` 指定的是即将运行的设备，这里需要传入的是 gpu ，通过指定该参数，PaddleSeg 调用飞桨的设备指定接口 `paddle.set_device` 来指定运行设备为 gpu，在进行模型训练时，飞桨将自动调用 gpu 算子用于执行模型计算。关于设备指定的更多细节，可以参考官方 api [paddle.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html#set-device)。

* 参数 `--config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml` 表示读取指定目录下的配置文件，配置文件中指定了模型结构，训练超参等所有训练模型需要用到的配置，该文件中指定的模型结构为 DeepLabv3+

```shell
python -u -m paddle.distributed.launch --devices 0,1,2,3 tools/train.py \
    --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
    --num_workers 8 \
    --save_dir output/deeplabv3p_resnet50 \
    --log_iters 10 \
    --device gpu \
    --do_eval \
    --save_interval 1000 \
    --seed 2048
```

上述命令会在 PaddleSeg 目录下产生一个 output/deeplabv3p_resnet50 目录，该目录会存放训练过程中的模型参数

## 四、模型导出 & 推理

### 模型导出

训练完成后，最优指标对应的权重放在 output/deeplabv3p_resnet50/best_model 目录下，执行以下命令将模型转成 Paddle 静态图格式存储，以获得更好的推理性能：

* export.py 执行的是 `动转静` 操作，飞桨框架会对代码进行分析，将动态图代码（灵活易用）转为 静态图模型（高效），以达到更加高效的推理性能

* 该操作会在指定 output/deeplabv3p_resnet50_inference_model 下生成 inference.pdiparams、inference.pdiparams.info、inference.pdmodel 3 个文件

```shell
python tools/export.py \
    --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
    --model_path output/deeplabv3p_resnet50/best_model/model.pdparams \
    --save_dir output/deeplabv3p_resnet50_inference_model
```

### 基于 PaddleInference 推理

推理代码位于 PaddleSeg/deploy 目录下，执行下列命令进行 DCU 推理：

* 该脚本将会加载上一步保存的静态图，使用飞桨预测库 PaddleInference 进行推理

* PaddleInference 内置了大量的高性能 Kernel，并且可以基于计算图分析，完成细粒度 OP 横向纵向融合，实现了高性能推理

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

export FLAGS_conv_workspace_size_limit=2000
python deploy/python/infer.py \
    --config output/deeplabv3p_resnet50_inference_model/deploy.yaml \
    --image_path ./cityscapes_demo.png \
    --save_dir ./output \
    --device "gpu"
```
