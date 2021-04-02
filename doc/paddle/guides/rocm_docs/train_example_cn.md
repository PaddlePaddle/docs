# 飞桨框架ROCM版训练示例

使用海光CPU与DCU进行训练与Nvidia GPU相同，当前Paddle ROCM版本完全兼容Paddle CUDA版本的API，直接使用原有的GPU训练命令和参数即可。

#### YoloV3训练示例

**第一步**：下载 YoloV3 代码

```bash
cd path_to_clone_PaddleDetection
git clone -b release/2.0-rc https://github.com/PaddlePaddle/PaddleDetection.git
```
也可以访问PaddleDetection的 [Github Repo](https://github.com/PaddlePaddle/PaddleDetection) 直接下载源码。

**第二步**：准备 VOC 数据集

```bash
cd PaddleDetection/dygraph/dataset/voc
python download_voc.py
python create_list.py
```

**第三步**：修改config文件的参数

模型Config文件 `dygraph/configs/yolov3/yolov3_darknet53_270e_voc.yml` 中的默认参数为8卡设计，使用DCU 4卡训练需要修改参数如下：

```bash
# 修改 dygraph/configs/yolov3/_base_/optimizer_270e.yml
base_lr: 0.0005

# 修改 dygraph/configs/yolov3/_base_/yolov3_reader.yml
worker_num: 1
```

**第四步**：运行训练

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

cd PaddleDetection/dygraph/
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/yolov3/yolov3_darknet53_270e_voc.yml --eval
```

**第五步**：获取4卡训练得到的mAP结果如下

```bash
# CUDA 结果为 CUDA 10.1 + 4卡V100 训练
CUDA - [03/23 05:26:17] ppdet.metrics.metrics INFO: mAP(0.50, 11point) = 82.59%

# ROCM 结果为 ROCM 4.0.1 + 4卡DCU 训练
ROCM - [03/28 16:02:52] ppdet.metrics.metrics INFO: mAP(0.50, 11point) = 83.02%
```
