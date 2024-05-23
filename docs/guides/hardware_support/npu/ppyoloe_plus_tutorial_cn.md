# 一、环境准备

## 环境说明

* 本教程介绍如何基于昇腾 910B NPU 进行 PP-YOLOE+的训练，总共需要 8 卡进行训练

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

3. 安装 PaddleDetection 代码库

```shell
git clone https://github.com/PaddlePaddle/PaddleDetection.git -b release_2_7_npu
cd PaddleDetection
python -m pip install -r requirements.txt
python -m pip install -e .
```

# 二、数据准备

请根据 [数据说明文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/docs/tutorials/PrepareDataSet.md#coco%E6%95%B0%E6%8D%AE) 准备 COCO 2017 数据集，准备完成后解压到 PaddleDetection/dataset/目录下，目录结构如下：

```
PaddleDetection/dataset/coco/
├── annotations
│   ├── instances_train2014.json
│   ├── instances_train2017.json
│   ├── instances_val2014.json
│   ├── instances_val2017.json
│   │   ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000580008.jpg
│   │   ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   │   ...
```

# 三、模型训练

进入 PaddleDetection 目录下，执行如下命令启动 8 卡 NPU（0 ~ 7 号卡）训练，其中：

* 参数 `--config configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml` 表示读取指定目录下的配置文件，配置文件中指定了模型结构，训练超参等所有训练模型需要用到的配置，该文件中指定的模型结构为 `PP-YOLOE+-l`

```shell
export FLAGS_npu_jit_compile=0
export FLAGS_use_stride_kernel=0

python -u -m paddle.distributed.launch --devices 0,1,2,3,4,5,6,7 \
    tools/train.py --eval --config configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml \
    --enable_ce True
```

上述命令会在 PaddleDetection 目录下产生一个 output/ppyoloe_plus_crn_l_80e_coco 目录，该目录会存放训练过程中的模型参数

# 四、模型导出 & 推理

## 模型导出

训练完成后，最优指标对应的权重放在 output/ppyoloe_plus_crn_l_80e_coco/pipeline/best_model/ 目录下，执行以下命令将模型转成 Paddle 静态图格式存储，以获得更好的推理性能：

* export_model.py 执行的是 动转静 操作，飞桨框架会对代码进行分析，将动态图代码（灵活易用）转为 静态图模型（高效），以达到更加高效的推理性能

* 该操作会在指定 output_inference/ppyoloe_plus_crn_l_80e_coco 下生成 inference.pdiparams、inference.pdiparams.info、inference.pdmodel 3 个文件

```shell
python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml -o weights=output/ppyoloe_plus_crn_l_80e_coco/pipeline/best_model/model.pdparams
```

## 基于 PaddleInference 推理

推理代码位于 PaddleDetection/deploy 目录下，执行下列命令进行 NPU 推理：

* 该脚本将会加载上一步保存的静态图，使用飞桨预测库 PaddleInference 进行推理

* PaddleInference 内置了大量的高性能 Kernel，并且可以基于计算图分析，完成细粒度 OP 横向纵向融合，实现了高性能推理

```shell
python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_crn_l_80e_coco --image_file=demo/000000014439_640x640.jpg --run_mode=paddle --device=npu
```
