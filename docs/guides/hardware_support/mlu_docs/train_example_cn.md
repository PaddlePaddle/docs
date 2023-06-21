# 飞桨框架 MLU 版训练示例

使用寒武纪 MLU370 进行训练与使用 Intel CPU/Nvidia GPU 训练相同，当前 Paddle MLU 版本完全兼容 Paddle CUDA 版本的 API，直接使用原有的 GPU 训练命令和参数即可。

#### ResNet50 训练示例

**第一步**：安装 MLU 支持的 Paddlepaddle

Paddle MLU 版的 Python 预测库请参考 [飞桨框架 MLU 版安装说明](./paddle_install_cn.html) 进行安装或编译。


**第二步**：下载 ResNet50 代码，并准备 ImageNet1k 数据集

```bash
cd path_to_clone_PaddleClas
git clone https://github.com/PaddlePaddle/PaddleClas.git
```
也可以访问 PaddleClas 的 [GitHub Repo](https://github.com/PaddlePaddle/PaddleClas) 直接下载源码。请根据[数据说明](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/data_preparation/classification_dataset.md)文档准备 ImageNet1k 数据集。

**第三步**：运行训练

使用飞浆 PaddleXXX 套件运行 MLU 可以通过设置 Global.device 参数为 mlu 来指定设备，其他模型也可以参考该使用方式

```bash
export MLU_VISIBLE_DEVICES=0,1,2,3

cd PaddleClas/
PADDLE_XCCL_BACKEND=mlu FLAGS_selected_mlus=0,1,2,3 \
python3.7 -m paddle.distributed.launch --devices="0,1,2,3" tools/train.py \
          -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
          -o Global.device=mlu
```
