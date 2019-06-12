# C++ 预测 API介绍

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

高层 API 底层有多种优化实现，我们称之为 engine；不同 engine 的切换通过传递不同的 Config 实现重载。

`Config` 有两种，`NativeConfig` 较简单和稳定，`AnalysisConfig` 功能更新，性能更好

- `NativeConfig` 原生 engine，由 paddle 原生的 forward operator
  组成，可以天然支持所有paddle 训练出的模型
  
- `AnalysisConfig` 
  - 支持计算图的分析和优化
  - 支持最新的各类 op fuse，性能一般比  `NativeConfig` 要好
  - 支持 TensorRT mixed engine 用于 GPU
    加速，用子图的方式支持了 [TensorRT] ，支持所有paddle
    模型，并自动切割部分计算子图到 TensorRT 上加速，具体的使用方式可以参考[这里](http://paddlepaddle.org/documentation/docs/zh/1.1/user_guides/howto/inference/paddle_tensorrt_infer.html)

## 基于 NativeConfig 的预测部署过程

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

### 基于 AnalysisConfig  提升性能

`AnalysisConfig` 是目前我们重点优化的版本。

类似 `NativeConfig` ， `AnalysisConfig` 可以创建一个经过一系列优化的高性能预测引擎。 其中包含了计算图的分析和优化，以及对一些重要 Op 的融合改写等，比如对使用了 While, LSTM, GRU 等模型性能有大幅提升 。

`AnalysisConfig` 的使用方法也和 `NativeConfig` 类似

```c++
AnalysisConfig config(dirname);  // dirname 是模型的路径
// 对于不同的模型存储格式，也可以用 AnalysisConfig config(model_file, params_file)
config.EnableUseGpu(100/*初始显存池大小(MB)*/, 0 /*gpu id*/);  // 使用GPU， CPU下使用config.DisableGpu();
config.SwitchIrOptim();                  // 打开优化开关，运行时会执行一系列的计算图优化
```

这里需要注意的是，输入的 PaddleTensor 需要指定，比如之前的例子需要修改为

```c++
auto predictor = paddle::CreatePaddlePredictor(config); // 注意这里需要 AnalysisConfig
// 创建输入 tensor
int64_t data[4] = {1, 2, 3, 4};
paddle::PaddleTensor tensor;
tensor.shape = std::vector<int>({4, 1});
tensor.data.Reset(data, sizeof(data));
tensor.dtype = paddle::PaddleDType::INT64;
```

后续的执行过程与 `NativeConfig` 完全一致。

### 变长序列输入
在处理变长输入的时候，需要对 `PaddleTensor` 设置LoD信息

``` c++
# 假设序列长度依次为 [3, 2, 4, 1, 2, 3]
tensor.lod = {{0,
               /*0 + 3=*/3,
               /*3 + 2=*/5,
               /*5 + 4=*/9,
               /*9 + 1=*/10,
               /*10 + 2=*/12,
               /*12 + 3=*/15}};
```

更详细的例子可以参考[LoD-Tensor使用说明](../../../user_guides/howto/basic_concept/lod_tensor.html)

### 多线程预测的建议

#### 数据并行的服务

这种场景下，每个服务线程执行同一种模型，支持 CPU 和 GPU。

Paddle 并没有相关的接口支持，但用户可以简单组合得出，下面演示最简单的实现，用户最好参考具体应用场景做调整

```c++
auto main_predictor = paddle::CreatePaddlePredictor(config);

const int num_threads = 10;  // 假设有 10 个服务线程
std::vector<std::thread> threads;
std::vector<decl_type(main_predictor)> predictors;

// 最好初始化时把所有predictor都创建好
predictors.emplace_back(std::move(main_predictor));
for (int i = 1; i < num_threads; i++) {
    predictors.emplace_back(main_predictor->Clone());
}
// 创建线程并执行
for (int i = 0; i < num_threads; i++) {
    threads.emplace_back([i, &]{
        auto& predictor = predictors[i];
        // 执行
        CHECK(predictor->Run(...));
    });
}

// 结尾
for (auto& t : threads) {
    if (t.joinable()) t.join();
}

// 结束
```

#### 模型并行的服务

这种场景，使用多个线程/CPU核加速单个模型的预测，**目前只支持 CPU下使用 MKL/MKLDNN 的情况**。

使用 `AnalysisConfig` 的对应接口来设置底层科学计算库使用线程的数目，具体参考 [SetCpuMathLibraryNumThreads](https://github.com/PaddlePaddle/Paddle/blob/release/1.3/paddle/fluid/inference/api/paddle_analysis_config.h#L159)

```c++
config.SetCpuMathLibraryNumThreads(8); // 一个模型使用 8 个线程加速预测

// 查询状态，可以使用如下接口
config.cpu_math_library_num_threads(); // return an int
```

### 性能建议

1. 在 CPU型号允许的情况下，尽量使用带 AVX 和 MKL 的版本
2. 复用输入和输出的 `PaddleTensor` 以避免频繁分配内存拉低性能
3. CPU或GPU预测，可以尝试把 `NativeConfig` 改成成 `AnalysisConfig` 来进行优化

#### CPU下可以尝试使用 Intel 的  `MKLDNN` 加速

MKLDNN 对 `CNN` 类的模型预测有不错的加速效果，可以尝试对比与 `MKLML` 的性能。

使用方法：

```c++
// AnalysisConfig config(...);
config.EnableMKLDNN();
// 查看 mkldnn 是否已经打开，可以用如下代码
config.mkldnn_enabled();  // return a bool
```

#### GPU 下可以尝试打开 `TensorRT` 子图加速引擎

通过计算图分析，Paddle 可以自动将计算图中部分子图切割，并调用 NVidia 的 `TensorRT` 来进行加速。

详细内容可以参考 [TensorRT 子图引擎](./paddle_tensorrt_infer.html)

## 详细代码参考

`AnalysisConfig` 完整接口可以参考 [这里](https://github.com/PaddlePaddle/Paddle/blob/release/1.3/paddle/fluid/inference/api/paddle_analysis_config.h#L35)

[inference demos](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/inference/api/demo_ci)


