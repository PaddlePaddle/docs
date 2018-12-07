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
| 1 | 8.53945 | 8.18737 |
| 2 | 14.2269 | 13.8976 |
| 4 | 24.2803 | 21.7976 |
| 8 | 45.6003 | 40.319 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1053.88 | 762.73 |
| 2 | 1055.71 | 762.41 |
| 4 | 1003.22 | 832.75 |
| 8 | 1108.77 | 926.9 |


### <span id = '2'>Yolo </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 8.41606| 7.07977 |
| 2 | 16.6588| 15.2216 |
| 4 | 31.9955| 30.5102 |
| 8 | 66.1107 | 64.3658 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1054.71  | 299.8 |
| 2 | 951.51  | 347.47 |
| 4 | 846.9  | 438.47 |
| 8 | 1042.31  | 515.15 |

### <span id = '3'> Resnet50 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 4.10063  |  3.33845 |
| 2 |  6.10941 |  5.54814 |
| 4 | 9.90233  | 10.2763 |
| 8 | 17.3287  |   20.0783 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1059.15 | 299.86 |
| 2 | 1077.8  | 340.78 |
| 4 | 903.04  | 395 |
| 8 | 832.53  | 508.86 |

### <span id = '4'> Resnet101 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 7.29828 | 5.672 |
| 2 | 11.2037 | 9.42352 |
| 4 | 17.9306 | 18.0936 |
| 8 | 31.4804 | 35.7439 |

- GPU Memory Used (`MB)`

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1161.94 | 429.22 |
| 2 | 1190.92 | 531.92 |
| 4 | 994.11  | 549.7 |
| 8 | 945.47  | 653.06 |

###  <span id = '5'> MobileNet V1 </span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1.52692  |  1.39282 |
| 2 |  1.98091  |  2.05788 |
| 4 | 3.2705  | 4.03476 |
| 8 |  5.15652 |  7.06651 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1144.35   | 99.6 |
| 2 | 1160.03    | 199.75 |
| 4 | 1098  | 184.33 |
| 8 | 990.71  | 232.11 |

###  <span id = '6'> MobileNet V2</span>

- Latency (`ms`) of different batch

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1.95961 | 1.78249 |
| 2 | 2.8709 | 3.01144 |
| 4 | 4.46131 | 5.43946 |
| 8 | 7.161 | 10.2081 |

- GPU Memory Used (`MB`)

| BatchSize | TensorRT | Anakin |
| --- | --- | --- |
| 1 | 1154.69 | 195.25 |
| 2 | 1187.25 | 227.6 |
| 4 | 1053 | 241.75 |
| 8 | 1062.48 | 352.18 |


## How to run those Benchmark models

1. 首先, 使用[External Converter](./convert_paddle_to_anakin.html)对caffe model 进行转换
2. 然后跳转至 *source_root/benchmark/CNN* 目录下，使用 'mkdir ./models'创建存放模型的目录，并将转换好的Anakin模型放在该目录下
3. 运行脚本 `sh run.sh`，运行结束后，该模型的运行时间将会显示到终端上
4. 如果你想获取每层OP的运行时间，你只用将 CMakeLists.txt 中的`ENABLE_OP_TIMER` 设置为 `YES` 即可
