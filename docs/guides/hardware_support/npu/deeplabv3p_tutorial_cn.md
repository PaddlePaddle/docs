# 一、环境准备

## 环境说明

* 本教程介绍如何基于昇腾 910B NPU 进行 DeepLabv3+的训练，总共需要 4 卡进行训练

* 考虑到环境差异性，我们推荐使用教程提供的标准镜像完成环境准备：

  * 镜像链接：registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py39

  * 镜像中已经默认安装了昇腾算子库 CANN-8.0.RC1

* 昇腾驱动版本为 23.0.3

## 环境安装

1. 安装 PaddlePaddle

*该命令会自动安装飞桨主框架每日自动构建的 nightly-build 版本*

```shell
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```

2. 安装 CustomDevice

*该命令会自动安装飞桨 Custom Device 每日自动构建的 nightly-build 版本*

```shell
python -m pip install paddle-custom-npu -i https://www.paddlepaddle.org.cn/packages/nightly/npu/
```

3. 安装 PaddleSeg 代码库

```shell
git clone https://github.com/PaddlePaddle/PaddleSeg -b npu_cann8.0
cd PaddleSeg
python -m pip install -r requirements.txt
python -m pip install -e .
```

# 二、数据准备

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

# 三、模型训练

进入 PaddleSeg 目录下，执行如下命令启动 4 卡 NPU（0 ~ 3 号卡）训练，其中：

* 参数 `--device` 指定的是即将运行的设备，这里需要传入的是 npu ，通过指定该参数，PaddleSeg 调用飞桨的设备指定接口 `paddle.set_device` 来指定运行设备为 npu，在进行模型训练时，飞桨将自动调用 npu 算子用于执行模型计算。关于设备指定的更多细节，可以参考官方 api [paddle.set_device](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/device/set_device_cn.html#set-device)。

* 参数 `--config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml` 表示读取指定目录下的配置文件，配置文件中指定了模型结构，训练超参等所有训练模型需要用到的配置，该文件中指定的模型结构为 DeepLabv3+

```shell
python -u -m paddle.distributed.launch --devices 0,1,2,3 tools/train.py \
    --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
    --num_workers 8 \
    --save_dir output/deeplabv3p_resnet50 \
    --log_iters 10 \
    --device npu \
    --do_eval \
    --save_interval 1000 \
    --seed 2048
```

上述命令会在 PaddleSeg 目录下产生一个 output/deeplabv3p_resnet50 目录，该目录会存放训练过程中的模型参数

# 四、模型导出 & 推理

## 模型导出

训练完成后，最优指标对应的权重放在 output/deeplabv3p_resnet50/best_model 目录下，执行以下命令将模型转成 Paddle 静态图格式存储，以获得更好的推理性能：

* export.py 执行的是 `动转静` 操作，飞桨框架会对代码进行分析，将动态图代码（灵活易用）转为 静态图模型（高效），以达到更加高效的推理性能

* 该操作会在指定 output/deeplabv3p_resnet50_inference_model 下生成 inference.pdiparams、inference.pdiparams.info、inference.pdmodel3 个文件

```shell
python tools/export.py \
    --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
    --model_path output/deeplabv3p_resnet50/best_model/model.pdparams \
    --save_dir output/deeplabv3p_resnet50_inference_model
```

## 基于 PaddleInference 推理

推理代码位于 PaddleSeg/deploy 目录下，执行下列命令进行 NPU 推理：

* 该脚本将会加载上一步保存的静态图，使用飞桨预测库 PaddleInference 进行推理

* PaddleInference 内置了大量的高性能 Kernel，并且可以基于计算图分析，完成细粒度 OP 横向纵向融合，实现了高性能推理

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

python deploy/python/infer.py \
    --config output/deeplabv3p_resnet50_inference_model/deploy.yaml \
    --image_path ./cityscapes_demo.png \
    --save_dir ./output \
    --device "npu"
```
