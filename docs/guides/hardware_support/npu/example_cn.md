# 昇腾 NPU 运行示例

**预先要求**：请先根据文档 [昇腾 NPU 安装说明](./install_cn.html) 准备昇腾 NPU 运行环境，建议以下步骤都在 docker 环境中运行。

## 训练示例

以 [ResNet50_vd](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/quick_start/quick_start_classification_new_user.md) 模型为例，介绍如何使用昇腾 NPU 进行训练。

### 一、下载套件代码

```bash
# 下载套件源码
git clone https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas/

# 安装 Python 依赖库
pip install -r requirements.txt

# 编译安装 paddleclas
python setup.py install
```

### 二、准备训练数据

进入 `PaddleClas/dataset` 目录，下载并解压 `flowers102` 数据集：

```bash
# 准备数据集 - 将数据集下载到对应的目录下，并解压
cd PaddleClas/dataset
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/flowers102.zip
unzip flowers102.zip

# 下载解压完成之后，当前目录结构如下
PaddleClas/dataset/flowers102
├── flowers102_label_list.txt
├── jpg
├── train_extra_list.txt
├── train_list.txt
└── val_list.txt
```

### 三、运行四卡训练

```bash
# 进入套件目录
cd PaddleClas/

# 四卡训练
python -m paddle.distributed.launch --devices "0,1,2,3" \
       tools/train.py -c ./ppcls/configs/quick_start/ResNet50_vd.yaml \
       -o Arch.pretrained=True \
       -o Global.device=npu
# 训练完成之后，预期得到输出如下
# ppcls INFO: [Eval][Epoch 20][best metric: 0.9215686917304993]
# ppcls INFO: Already save model in ./output/ResNet50_vd/epoch_20
# ppcls INFO: Already save model in ./output/ResNet50_vd/latest

# 单卡评估 - 使用上一步训练得到的模型进行评估
python tools/eval.py -c ./ppcls/configs/quick_start/ResNet50_vd.yaml \
       -o Arch.pretrained="output/ResNet50_vd/best_model" \
       -o Global.device=npu
# 评估完成之后，预期得到输出如下
# [Eval][Epoch 0][Avg]CELoss: 0.45636, loss: 0.45636, top1: 0.91961, top5: 0.98725
```

## 推理示例

以 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型为例，介绍如何使用昇腾 NPU 进行推理。

### 一、下载推理程序

```bash
# 下载 Paddle-Inference-Demo 示例代码，并进入 Python 代码目录
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git
```

### 二、准备推理模型

```bash
# 进入 python npu 推理示例程序目录
cd Paddle-Inference-Demo/python/npu/resnet50

# 下载推理模型文件并解压
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 准备预测示例图片
wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg

# 准备完成后的模型和图片目录如下
Paddle-Inference-Demo/python/npu/resnet50
├── ILSVRC2012_val_00000247.jpeg
└── resnet50
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

### 三、运行推理程序

```bash
# 运行 Python 推理程序
python infer_resnet.py \
    --model_file=./resnet50/inference.pdmodel \
    --params_file=./resnet50/inference.pdiparams

# 预期得到输出如下
# class index:  13
```
