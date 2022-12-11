如何进行基准测试
===============
本文介绍如何给深度学习框架做基准测试。基准测试主要包含验证模型的精度和性能两方面，下文包含搭建测试环境，选择基准测试模型，验证测试结果等几方面内容。

验证深度学习框架，可分为训练和测试两个阶段， 验证指标略有不同，本文只介绍训练阶段的指标验证。训练阶段关注的是模型训练集上的精度，训练集是完备的，因此关注大 batch\_size 下的训练速度,关注吞吐量，例如图像模型常用的 batch\_size=128, 多卡情况下会加大；预测阶段关注的是在测试集上的精度，线上服务测试数据不能提前收集，因此关注小 batch\_size 下的预测速度，关注延迟，例如预测服务常用的 batch\_size=1, 4 等。

[Fluid](https://github.com/PaddlePaddle/Paddle>)是 PaddlePaddle 从 0.11.0 版本开始引入的设计，本文的基准测试在该版本上完成。


环境搭建
========

基准测试中模型精度和硬件、框架无关，由模型结构和数据共同决定；性能方面由测试硬件和框架性能决定。框架基准测试为了对比框架之间的差异，控制硬件环境，系统库等版本一致。下文中的对比实验都在相同的硬件条件和系统环境条件下进行.


不同架构的 GPU 卡性能差异巨大，在验证模型在 GPU 上训练性能时，可使用 NVIDIA 提供的命令:```nvidia-smi``` 检验当前使用的 GPU 型号，如果测试多卡训练性能，需确认硬件连接是 [nvlink](https://zh.wikipedia.org/zh/NVLink)或 [PCIe](https://zh.wikipedia.org/zh-hans/PCI_Express)。 同样地，CPU 型号会极大影响模型在 CPU 上的训练性能。可读取`/proc/cpuinfo`中的参数，确认当前正在使用的 CPU 型号。

下载 GPU 对应的 Cuda Tool Kit 和 Cudnn，或者使用 NVIDIA 官方发布的 nvidia-docker 镜像 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), 镜像内包含了 Cuda 和 Cudnn，本文采用这种方式。 Cuda Tool Kit 包含了 GPU 代码使用到的基础库，影响在此基础上编译出的 Fluid 二进制运行性能。

准备好 Cuda 环境后，从 github 上下载 Paddle 代码并编译，会生成对应的最适合当前 GPU 的 sm\_arch 二进制[sm\_arch](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。另外，cudnn 对卷积类任务影响巨大，在基准测试中需要小版本一致，例如 Cudnn7.0.2 与 Cudnn7.1.4 在 Resnet 上有 5%以上差异。


选择基准模型
============

对框架做基准测试，需要覆盖不同训练任务和不同大小的模型，本文中选取了图像和 NLP 的最为常用的 5 个模型。

任务种类|        模型名称|       网络结构|         数据集
:---:|:--:|:---:|:---:
图像生成|      CycleGAN|         GAN|              horse2zebra
图像分类|      SE-ResNeXt50|        Resnet-50|          image-net
语义分割|      DeepLab_V3+|  ResNets|       cityscapes
自然语言|      Bert|       Transformer|       Wikipedia
机器翻译|      Transformer|           Attention|             Wikipedia

CycleGAN, SE-ResNeXt50, DeepLab_V3+属于 CNN 模型, Bert, Transformer 是一种比传统 RNN 模型更好的 NLP 模型。
[benchmark](https://github.com/PaddlePaddle/Paddle/tree/develop/benchmark/fluid)
基准模型测试脚本中，均跳过了前几个 batch 的训练过程，原因是加载数据和分配显存受系统当前运行情况影响，会导致统计性能不准确。运行完若干个轮次后，统计对应指标。


基准模型的数据的选择方面，数据量大且验证效果多的公开数据集为首选。图像模型 CycleGAN 选择了 horse2zebra 数据集，SE-ResNeXt50 选择了[image-net](http://www.image-net.org/challenges/LSVRC/2012/nnoupb)数据集，图像大小预处理为和 Imagenet 相同大小，因此性能可直接对比。
NLP 模型的公开且影响力大数据集较少，Bert 和 Transformer 模型都选择了[Wikipedia](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2)数据集。


注意，图像模型每条样本大小相同，图像经过变换后大小一致，因此经过的计算路径基本相同，计算速度和显存占用波动较小，可以从若干个 batch 的数据中采样得到当前的训练性能数据。而 NLP 模型由于样本长度不定，计算路径和显存占用也不相同，因此只能完整运行若干个轮次后，统计速度和显存消耗。
显存分配是特别耗时的操作，因此 Fluid 默认会占用所有可用显存空间形成显存池，用以加速计算过程中的显存分配。如果需要统计模型真实显存消耗，可设置环境变量`FLAGS_fraction_of_gpu_memory_to_use=0.0`，观察最大显存开销。


测试过程
========

-  GPU 单机单卡测试

本教程使用了 Cuda9, Cudnn7.0.1。来源为:```nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04```

```
    nvidia-docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmark -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu paddlepaddle/paddle:latest-dev /bin/bash
```
在单卡上测试，设置 CUDA 的环境变量使用一块 GPU，``CUDA_VISIBLE_DEVICES=0``
然后代码中设置为使用 CUDAPlace，如果使用 Paddle 代码库中的脚本，只需要命令行参数传入 use_gpu=True 即可。

```
    >>> import paddle.fluid as fluid
    >>> place = fluid.CUDAPlace(0) // 0 指第 0 块 GPU
```

测试结果
========

本教程对比相同环境下的 Fluid1.4, Pytorch1.1.0 和 TensorFlow1.12.0 的性能表现。
硬件环境为 CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz, GPU: Tesla v100(volta) 21729MiB x 1, Nvidia-Driver 384.66。
系统环境为 Ubuntu 16.04.3 LTS, 本文中采用了 docker 环境，系统版本为 nvidia-docker17.05.0-ce。
测试的 Fluid 版本为[v.1.4.1](https://github.com/PaddlePaddle/Paddle/tree/v1.4.1) 。
TensorFlow 版本为[v.1.12.0-rc2](https://github.com/tensorflow/tensorflow/tree/v1.12.0-rc2)。
PyTorch 版本为[v.1.1.0](https://github.com/pytorch/pytorch/tree/v1.1.0)。
使用的脚本和配置见[benchmark](https://github.com/PaddlePaddle/Paddle/tree/develop/benchmark/fluid) 。
SE-ResNeXt50 对比的框架是 PyTorch，因为 TensorFlow 上没有对应的模型。
图表中统计单位为 samples/秒。



- GPU 单机单卡测试结果

  Model|Fluid GPU|  TensorFlow/PyTorch GPU
  :---:|:--:|:---:
  CycleGAN|              7.3 samples/s|               6.1 samples/s
  SE-ResNeXt50|             169.4 samples/s  |              153.1 samples/s
  DeepLab_V3+|          12.8 samples/s  |              6.4 samples/s
  Bert|       4.0 samples/s   |              3.4 samples/s
  Transformer|            4.9 samples/s   |              4.7 samples/s
