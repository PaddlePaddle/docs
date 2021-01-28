# 飞桨框架昆仑XPU版训练示例

使用XPU训练与cpu/gpu相同，只需要加上-o use_xpu=True, 表示执行在昆仑设备上。

#### ResNet50下载并运行示例：

模型文件下载命令：

```
cd path_to_clone_PaddleClas
git clone -b release/static https://github.com/PaddlePaddle/PaddleClas.git
```
也可以访问PaddleClas的[github repo](https://github.com/PaddlePaddle/PaddleClas/tree/release/static)直接下载源码。

运行训练：
```
#FLAGS指定单卡或多卡训练，此示例运行2个卡
export FLAGS_selected_xpus=0,1
#启动训练
Python3.7 tools/train_multi_platform.py -c configs/kunlun/ResNet50.yaml -o use_gpu=False -o use_xpu=True
```

注意：飞腾CPU+昆仑XPU的环境下暂未支持多卡训练。
