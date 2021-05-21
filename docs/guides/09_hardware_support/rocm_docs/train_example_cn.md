# 飞桨框架ROCm版训练示例

使用海光CPU与DCU进行训练与Nvidia GPU相同，当前Paddle ROCm版本完全兼容Paddle CUDA版本的API，直接使用原有的GPU训练命令和参数即可。

#### ResNet50训练示例

**第一步**：下载 ResNet50 代码，并准备 ImageNet1k 数据集

```bash
cd path_to_clone_PaddleClas
git clone https://github.com/PaddlePaddle/PaddleClas.git
```
也可以访问PaddleClas的 [Github Repo](https://github.com/PaddlePaddle/PaddleClas) 直接下载源码。请根据[数据说明](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.0/docs/zh_CN/tutorials/data.md)文档准备ImageNet1k数据集。

**第二步**：运行训练

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

cd PaddleClas/
python -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ./configs/ResNet/ResNet50.yaml
```

**第三步**：获取4卡训练得到的 Best Top1 Accuracy 结果如下

```bash
# CUDA 结果为 CUDA 10.1 + 4卡V100 训练
2021-03-24 01:16:08,548 - INFO - The best top1 acc 0.76332, in epoch: 118

# ROCm 结果为 ROCm 4.0.1 + 4卡DCU 训练
2021-04-07 10:26:31,651 - INFO - The best top1 acc 0.76308, in epoch: 109
```

#### YoloV3训练示例

**第一步**：下载 YoloV3 代码

```bash
cd path_to_clone_PaddleDetection
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```
也可以访问PaddleDetection的 [Github Repo](https://github.com/PaddlePaddle/PaddleDetection) 直接下载源码。

**第二步**：准备 VOC 数据集

```bash
cd PaddleDetection/dataset/voc
python download_voc.py
python create_list.py
```

**第三步**：修改config文件的参数

模型Config文件 `configs/yolov3/yolov3_darknet53_270e_voc.yml` 中的默认参数为8卡设计，使用DCU单机4卡训练需要修改参数如下：

```bash
# 修改 configs/yolov3/_base_/optimizer_270e.yml
base_lr: 0.0005

# 修改 configs/yolov3/_base_/yolov3_reader.yml
worker_num: 1
```

**第四步**：运行训练

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

cd PaddleDetection/
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/yolov3/yolov3_darknet53_270e_voc.yml --eval
```

**第五步**：获取4卡训练得到的 mAP 结果如下

```bash
# CUDA 结果为 CUDA 10.1 + 4卡V100 训练
[03/23 05:26:17] ppdet.metrics.metrics INFO: mAP(0.50, 11point) = 82.59%

# ROCm 结果为 ROCm 4.0.1 + 4卡DCU 训练
[03/28 16:02:52] ppdet.metrics.metrics INFO: mAP(0.50, 11point) = 83.02%
```
