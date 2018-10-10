# Anakin GPU 性能测试

## 环境:

>  CPU: `12-core Intel(R) Xeon(R) CPU E5-2620 v2 @2.10GHz`
>  GPU: `Tesla P4`
>  cuDNN: `v7`


## anakin 对比对象:

**`Anakin`** 将与高性能的推理引擎 **`NVIDIA TensorRT 3`** 进行比较

## Benchmark Model

> 注意在性能测试之前，请先将测试model通过 `External Converter` 工具转换为Anakin model
> 对这些model，本文在GPU上进行单线程单GPU卡的性能测试。

- [Vgg16](#1)   *caffe model 可以在[这儿](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395)下载*
- [Yolo](#2)  *caffe model 可以在[这儿](https://github.com/hojel/caffe-yolo-model)下载*
- [Resnet50](#3)  *caffe model 可以在[这儿](https://github.com/KaimingHe/deep-residual-networks#models)下载*
- [Resnet101](#4)  *caffe model 可以在[这儿](https://github.com/KaimingHe/deep-residual-networks#models)下载*
- [Mobilenet v1](#5)  *caffe model 可以在[这儿](https://github.com/shicai/MobileNet-Caffe)下载*
- [Mobilenet v2](#6)  *caffe model 可以在[这儿](https://github.com/shicai/MobileNet-Caffe)下载*
- [RNN](#7)  *暂不支持*

### <span id = '1'>VGG16 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 8.8690 | 8.2815 |
| 2 | 15.5344 | 13.9116 |
| 4 | 26.6000 | 21.8747 |
| 8 | 49.8279 | 40.4076 |
| 32 | 188.6270 | 163.7660 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 963 | 997 |
| 2 | 965 | 1039 |
| 4 | 991 | 1115 |
| 8 | 1067 | 1269 |
| 32 | 1715 | 2193 |


### <span id = '2'>Yolo </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 16.4596| 15.2124 |
| 2 | 26.6347| 25.0442 |
| 4 | 43.3695| 43.5017 |
| 8 | 80.9139 | 80.9880 |
| 32 | 293.8080| 310.8810 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1569 | 1775 |
| 2 | 1649 | 1815 |
| 4 | 1709 | 1887 |
| 8 | 1731 | 2031 |
| 32 | 2253 | 2907 |

### <span id = '3'> Resnet50 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 4.2459   |  4.1061 |
| 2 |  6.2627  |  6.5159 |
| 4 | 10.1277  | 11.3327 |
| 8 | 17.8209  | 20.6680 |
| 32 | 65.8582 | 77.8858 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 531  | 503 |
| 2 | 543  | 517 |
| 4 | 583 | 541 |
| 8 | 611 | 589 |
| 32 |  809 | 879 |

### <span id = '4'> Resnet101 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 7.5562 | 7.0837 |
| 2 | 11.6023 | 11.4079 |
| 4 | 18.3650 | 20.0493 |
| 8 | 32.7632 | 36.0648 |
| 32 | 123.2550 | 135.4880 |

- GPU Memory Used (`MB)`

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 701  | 683 |
| 2 | 713  | 697 |
| 4 | 793 | 721 |
| 8 | 819 | 769 |
| 32 | 1043 | 1059 |

###  <span id = '5'> MobileNet V1 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 45.5156  |  1.3947 |
| 2 |  46.5585  |  2.5483 |
| 4 | 48.4242  | 4.3404 |
| 8 |  52.7957 |  8.1513 |
| 32 | 83.2519 | 31.3178 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 329  | 283 |
| 2 | 345  | 289 |
| 4 | 371 | 299 |
| 8 | 393 | 319 |
| 32 |  531 | 433 |

###  <span id = '6'> MobileNet V2</span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 65.6861 | 2.9842 |
| 2 | 66.6814 | 4.7472 |
| 4 | 69.7114 | 7.4163 |
| 8 | 76.1092 | 12.8779 |
| 32 | 124.9810 | 47.2142 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 341 | 293 |
| 2 | 353 | 301 |
| 4 | 385 | 319 |
| 8 | 421 | 351 |
| 32 | 637 | 551 |

## How to run those Benchmark models

1. 首先, 使用[External Converter](./convert_paddle_to_anakin.html)对caffe model 进行转换
2. 然后跳转至 *source_root/benchmark/CNN* 目录下，使用 'mkdir ./models'创建存放模型的目录，并将转换好的Anakin模型放在该目录下
3. 运行脚本 `sh run.sh`，运行结束后，该模型的运行时间将会显示到终端上
4. 如果你想获取每层OP的运行时间，你只用将 CMakeLists.txt 中的`ENABLE_OP_TIMER` 设置为 `YES` 即可
