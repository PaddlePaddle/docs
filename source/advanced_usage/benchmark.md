## 概要
本文介绍如何给深度学习框架做benchmark. benchmark测试主要包含验证模型的精度和性能两方面, 下文通过搭建测试环境和测试指标等多方面对Fluid进行基准测试.
[Fluid](https://github.com/PaddlePaddle/Paddle) 是PaddlePaddle 从0.11.0 引入的新设计, 用来让用户像Pytorch和Tensorflow Eager Execution一样写程序. 
验证深度学习框架，可分为训练和测试两个阶段。验证指标大同小异，例如训练关注的是训练集上的精度，大batch_size下的训练速度，而测试关注的是在测试集上的精度，小batch_size下的预测速度。本文只介绍训练阶段的指标验证。

### 环境搭建
下文中的对比实验都在相同的硬件条件和软件条件下进行. benchmark的结果强依赖于硬件。所以需要保证影响指标的硬件环境一致，这样benchmark才有意义。
对硬件方面，对GPU验证需要保证GPU的型号一致; 对cpu需要保证CPU型号一致，另对于厂内机器，需要额外关注内存插条顺序是最优的；
对软件方面，基础库Cuda大版本需要保持一致，例如Cuda9与Cuda8支持指令不同，运行速度有差异，在此软件环境下，源码编译paddle。paddle会针对cuda生成对应的sm_arch二进制[sm_arch](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。cudnn对重卷积类任务影响巨大，需要版本**完全一致**。
本教程中的硬件环境为 Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz. TITAN X (Pascal) 12G x 1
系统环境为 System: Ubuntu 16.04.3 LTS, Nvidia-Docker 17.05.0-ce, build 89658be. Nvidia-Driver 384.90.
采用的Fluid版本为[v.0.12.0](https://github.com/PaddlePaddle/Paddle/releases/tag/v.0.12.0), 需要commit一致.

### 模型选择
benchmark需要兼顾大小模型，不同训练任务下的表现, 才能说明框架效果。其中mnist, VGG, Resnet属于CNN模型, stacked-lstm代表RNN模型. [benchmark](https://github.com/PaddlePaddle/Paddle/tree/develop/benchmark/fluid)

### 测试过程

- CPU

首先需要屏蔽GPU `export CUDA_VISIBLE_DEVICES=`;

在单机单卡的测试环境中,Fluid需要关闭OpenMP和MKL的多线程. 设置`export OMP_NUM_THREADS=1;export MKL_NUM_THREADS=1;`. 
TensorFlow需要关闭多线程, 设置 intra_op_parallelism_threads=1, inter_op_parallelism_threads=1.
运行过程中可以通过 `nvidia-smi`来校验是否有GPU被使用, 下文GPU同理.

```bash
docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmarkIMAGE_NAME /bin/bash
```
将其中的CASE_NAME和IMAGE_NAME换为对应的名字，运行对应的例子

- GPU 

再次确认cudnn和cuda版本一致。本教程使用了cudnn7, cuda8. nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
```bash
nvidia-docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmark -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu IMAGE_NAME /bin/bash
```
将其中的CASE_NAME和IMAGE_NAME换为对应的名字，运行对应的例子

### 测试结果

CPU测试结果

| Speed        | Fluid  CPU       | TensorFlow CPU   |
| ------------ | ---------------- | ---------------- |
| mnist        | 46.198 s/pass    | 94.106 s/pass    |
| VGG-16       | 0.4147 images/s  | 0.1229 images/s  |
| Resnet-50    | 1.6935  images/s | 0.3657  images/s |
| Stacked-LSTM | 472.3225 words/s | 48.2293words/s   |
| Seq2Seq      | 217.1655 words/s | 28.6164 words/s  |

GPU测试结果

| Speed        | Fluid  GPU   | TensorFlow GPU    |
| ------------ | ------------ | ----------------- |
| mnist        | 3.044 s/pass | 3.852 s/pass      |
| VGG-16       | 59.83327     | 40.9967 images/s  |
| Resnet-50    | 105.84412    | 97.8923 images/s  |
| Stacked-LSTM | 1319.99315   | 1608.2526 words/s |
| Seq2Seq      | 7147.89081   | 6845.1161 words/s |

注：mnist由于图像太小，采用计量单位为s

### Reference

- PaddlePaddle Fluid [Paddle](https://github.com/PaddlePaddle/Paddle)

- TensorFlow [TensorFlow](https://github.com/tensorflow/tensorflow)
