#################
如何进行基准测试
#################

本文介绍如何给深度学习框架做基准测试。基准测试主要包含验证模型的精度和性能两方面，下文包含搭建测试环境, 选择基准测试模型, 验证测试结果等几方面内容。
验证深度学习框架，可分为训练和测试两个阶段, 验证指标略有不同，本文只介绍训练阶段的指标验证。训练阶段关注的是模型训练集上的精度，训练集是完备的，因此关注大batch\_size下的训练速度,关注吞吐量，例如图像模型常用的batch\_size=128, 多卡情况下会加大; 预测阶段关注的是在测试集上的精度，线上服务测试数据不能提前收集，因此关注小batch\_size下的预测速度，关注延迟，例如预测服务常用的batch\_size=1, 4等。
`Fluid <https://github.com/PaddlePaddle/Paddle>`__ 是PaddlePaddle从0.11.0版本开始引入的设计，本文的基准测试在该版本上完成。

环境搭建
########
基准测试中模型的精度和硬件、框架无关，由模型结构和数据共同决定; 性能方面由测试硬件和框架本身性能共同决定。为了对比框架之间的差异，需要控制硬件环境，系统库等版本一致。下文中的对比实验都在相同的硬件条件和系统环境条件下进行.
对硬件方面，对GPU验证需要保证GPU的型号一致; 对CPU需要保证CPU型号一致;
对软件方面，基础库Cuda大版本需要保持一致，例如Cuda9与Cuda8支持指令不同，运行速度有差异，在此软件环境下，源码编译paddle。paddle会针对cuda生成对应的sm\_arch二进制\ `sm\_arch <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`__\ 。cudnn对重卷积类任务影响巨大，需要版本\ **完全一致**\ 。
本教程中的硬件环境为 Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz. TITAN X
(Pascal) 12G x 1 系统环境为 System: Ubuntu 16.04.3 LTS, Nvidia-Docker
17.05.0-ce, build 89658be. Nvidia-Driver 384.90.
采用的Fluid版本为\ `v.0.12.0 <https://github.com/PaddlePaddle/Paddle/releases/tag/v.0.12.0>`__,
需要commit一致.

基准模型选择
############

benchmark需要兼顾大小模型，不同训练任务下的表现,
才能说明框架效果。其中mnist, VGG, Resnet属于CNN模型,
stacked-lstm代表RNN模型.
`benchmark <https://github.com/PaddlePaddle/Paddle/tree/develop/benchmark/fluid>`__

测试过程
########

-  CPU

首先需要屏蔽GPU ``export CUDA_VISIBLE_DEVICES=``;

在单机单卡的测试环境中,Fluid需要关闭OpenMP和MKL的多线程.
设置\ ``export OMP_NUM_THREADS=1;export MKL_NUM_THREADS=1;``.
TensorFlow需要关闭多线程, 设置 intra\_op\_parallelism\_threads=1,
inter\_op\_parallelism\_threads=1. 运行过程中可以通过
``nvidia-smi``\ 来校验是否有GPU被使用, 下文GPU同理.

.. code:: bash

    docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmarkIMAGE_NAME /bin/bash

将其中的CASE\_NAME和IMAGE\_NAME换为对应的名字，运行对应的例子

-  GPU

再次确认cudnn和cuda版本一致。本教程使用了cudnn7, cuda8.
nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

.. code:: bash

    nvidia-docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmark -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu IMAGE_NAME /bin/bash

将其中的CASE\_NAME和IMAGE\_NAME换为对应的名字，运行对应的例子

测试结果
########

CPU测试结果

+----------------+--------------------+-------------------+
| Speed          | Fluid CPU          | TensorFlow CPU    |
+================+====================+===================+
| mnist          | 46.198 s/pass      | 94.106 s/pass     |
+----------------+--------------------+-------------------+
| VGG-16         | 0.4147 images/s    | 0.1229 images/s   |
+----------------+--------------------+-------------------+
| Resnet-50      | 1.6935 images/s    | 0.3657 images/s   |
+----------------+--------------------+-------------------+
| Stacked-LSTM   | 472.3225 words/s   | 48.2293words/s    |
+----------------+--------------------+-------------------+
| Seq2Seq        | 217.1655 words/s   | 28.6164 words/s   |
+----------------+--------------------+-------------------+

GPU测试结果

+----------------+----------------+---------------------+
| Speed          | Fluid GPU      | TensorFlow GPU      |
+================+================+=====================+
| mnist          | 3.044 s/pass   | 3.852 s/pass        |
+----------------+----------------+---------------------+
| VGG-16         | 59.83327       | 40.9967 images/s    |
+----------------+----------------+---------------------+
| Resnet-50      | 105.84412      | 97.8923 images/s    |
+----------------+----------------+---------------------+
| Stacked-LSTM   | 1319.99315     | 1608.2526 words/s   |
+----------------+----------------+---------------------+
| Seq2Seq        | 7147.89081     | 6845.1161 words/s   |
+----------------+----------------+---------------------+

注：mnist由于图像太小，采用计量单位为s
