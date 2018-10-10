# Paddle 预测 API

为了更简单方便的预测部署，Fluid 提供了一套高层 API 用来隐藏底层不同的优化实现。

预测库包含:

- 头文件 `paddle_inference_api.h` 定义了所有的接口
- 库文件`libpaddle_fluid.so` 或 `libpaddle_fluid.a`

下面是详细介绍

## PaddleTensor

PaddleTensor 定义了预测最基本的输入输出的数据格式，常用字段：

- `name` 用于指定输入数据对应的 模型中variable 的名字
- `shape` 表示一个 Tensor 的 shape
- `data`  数据以连续内存的方式存储在`PaddleBuf` 中，`PaddleBuf` 可以接收外面的数据或者独立`malloc`内存，详细可以参考头文件中相关定义。
- `dtype` 表示 Tensor 的数据类型

## 利用Config 创建不同引擎

高层 API 底层有多种优化实现，我们称之为 engine；不同 engine 的切换通过传递不同的 Config 实现重载

- `NativeConfig` 原生 engine，由 paddle 原生的 forward operator
    组成，可以天然支持所有paddle 训练出的模型，

- `MixedRTConfig` TensorRT mixed engine 用于 GPU
    加速，用子图的方式支持了 [TensorRT] ，支持所有paddle
    模型，并自动切割部分计算子图到 TensorRT 上加速（WIP）


## 预测部署过程

总体上分为以下步骤

1. 用合适的配置创建 `PaddlePredictor`
2. 创建输入用的 `PaddleTensor`，传入到 `PaddlePredictor` 中
3. 获取输出的 `PaddleTensor` ，将结果取出

下面完整演示一个简单的模型，部分细节代码隐去

```c++
#include "paddle_inference_api.h"

// 创建一个 config，并修改相关设置
paddle::NativeConfig config;
config.model_dir = "xxx";
config.use_gpu = false;
// 创建一个原生的 PaddlePredictor
auto predictor =
      paddle::CreatePaddlePredictor<paddle::NativeConfig>(config);
// 创建输入 tensor
int64_t data[4] = {1, 2, 3, 4};
paddle::PaddleTensor tensor;
tensor.shape = std::vector<int>({4, 1});
tensor.data.Reset(data, sizeof(data));
tensor.dtype = paddle::PaddleDType::INT64;
// 创建输出 tensor，输出 tensor 的内存可以复用
std::vector<paddle::PaddleTensor> outputs;
// 执行预测
CHECK(predictor->Run(slots, &outputs));
// 获取 outputs ...
```

编译时，联编 `libpaddle_fluid.a/.so` 便可。 



## 高阶使用

### 输入输出的内存管理
`PaddleTensor` 的 `data` 字段是一个 `PaddleBuf`，用于管理一段内存用于数据的拷贝。 

`PaddleBuf` 在内存管理方面有两种模式：

1. 自动分配和管理内存

```c++
int some_size = 1024;
PaddleTensor tensor;
tensor.data.Resize(some_size);
```

2. 外部内存传入

```c++
int some_size = 1024;
// 用户外部分配内存并保证 PaddleTensor 使用过程中，内存一直可用
void* memory = new char[some_size]; 

tensor.data.Reset(memory, some_size);
// ...

// 用户最后需要自行删除内存以避免内存泄漏
delete[] memory;
```

两种模式中，第一种比较方便；第二种则可以严格控制内存的管理，便于与 `tcmalloc` 等库的集成。

### 基于 contrib::AnalysisConfig  提升性能 (预发布)
*AnalyisConfig 目前正在预发布阶段，用 `namespace contrib` 进行了保护，后续可能会有调整*

类似 `NativeConfig` ， `AnalysisConfig` 可以创建一个经过一系列优化的高性能预测引擎。 其中包含了计算图的分析和优化，以及对一些重要 Op 的融合改写等，**对使用了 While, LSTM, GRU 等模型性能有大幅提升** 。

`AnalysisConfig` 的使用方法也和 `NativeConfig` 类似，但 *目前仅支持 CPU，正在增加对GPU 的支持*

```c++
AnalysisConfig config;
config.model_dir = xxx;
config.use_gpu = false;  // 目前还不支持 GPU 的优化
config.specify_input_name = true; // 需要指定输入的 name
```

这里需要注意的是，输入的 PaddleTensor 需要指定，比如之前的例子需要修改为

```c++
auto predictor =
      paddle::CreatePaddlePredictor<paddle::contrib::AnalysisConfig>(config); // 注意这里需要 AnalysisConfig
// 创建输入 tensor
int64_t data[4] = {1, 2, 3, 4};
paddle::PaddleTensor tensor;
tensor.shape = std::vector<int>({4, 1});
tensor.data.Reset(data, sizeof(data));
tensor.dtype = paddle::PaddleDType::INT64;
tensor.name = "input0"; // 注意这里的 name 需要设定
```

### 性能建议
1. 在 CPU型号允许的情况下，尽量使用带 AVX 和 MKL 的版本
2. 复用输入和输出的 `PaddleTensor` 以避免频繁分配内存拉低性能
3. CPU预测，可以尝试把 `NativeConfig` 改成成 `AnalysisConfig` 来进行优化

## 详细代码参考

- [inference demos](./demo_ci)
- [复杂单线程/多线程例子](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/test_api_impl.cc)
