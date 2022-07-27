# 飞桨框架昆仑 2 代芯片训练示例

使用 XPU 训练与 CPU/GPU 相同，只需要简单配置 XPU，就可以执行在昆仑设备上。

#### ResNet50 下载并运行示例：

1、 安装依赖：
```
git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop
cd PaddleClas
python -m pip install -r requirements.txt
```

2、下载数据集：
基于 CIFAR100 数据集的 ResNet50 训练任务
```
cd dataset
rm -rf ILSVRC2012
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_CIFAR100.tar
tar xf whole_chain_CIFAR100.tar
ln -s whole_chain_CIFAR100 ILSVRC2012
cd ILSVRC2012
mv train.txt train_list.txt
mv test.txt val_list.txt
```

3、配置 XPU 进行训练的命令非常简单：
```
cd ../..
#FLAGS 指定单卡或多卡训练，此示例运行 1 个卡
export FLAGS_selected_xpus=2
export XPUSIM_DEVICE_MODEL=KUNLUN2
#启动训练
python tools/train.py \
-c ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
-o Global.device=xpu
```

#### YOLOv3-DarkNet53 下载并运行示例：

1、安装依赖：
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git -b develop
cd PaddleDetection/
pip install -U pip Cython
pip install -r requirements.txt
```

2、下载数据集
```
cd dataset/voc/
python download_voc.py
python create_list.py
cd ../..
```

3、配置 XPU 进行训练的命令非常简单：
```
#FLAGS 指定单卡或多卡训练，此示例运行 1 个卡
export FLAGS_selected_xpus=2
export XPUSIM_DEVICE_MODEL=KUNLUN2
#启动训练
python -u tools/train.py \
-c configs/yolov3/yolov3_darknet53_270e_voc.yml \
-o use_gpu=false use_xpu=true LearningRate.base_lr=0.000125 \
--eval
```
