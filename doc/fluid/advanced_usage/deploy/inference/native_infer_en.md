# Introduction to C++ Inference API

To make the deployment of inference model more convenient, a set of high-level APIs are provided in Fluid to hide diverse optimization processes in low level.

Inference library contains:

- header file `paddle_inference_api.h` which defines all interfaces
- library file `libpaddle_fluid.so` or `libpaddle_fluid.a`

Details are as follows:

## PaddleTensor

PaddleTensor defines basic format of input and output data for inference. Common fields are as follows:

- `name` is used to indicate the name of variable in model correspondent with input data.
- `shape` represents the shape of a Tensor.
- `data`  is stored in `PaddleBuf` in method of consecutive storage. `PaddleBuf` can receieve outer data or independently `malloc` memory. You can refer to associated definitions in head file.
- `dtype` represents data type of Tensor.

## Use Config to create different engines

The low level of high-level API contains various optimization methods which are called engines. Switch between different engines is done by transferring different Config.

- `NativeConfig` native engine, consisting of native forward operators of paddle, can naturally support all models trained by paddle.

- `AnalysisConfig` TensorRT mixed engine. It is used to speed up GPU and supports [TensorRT] with subgraph. Moreover, this engine supports all paddle models and automatically slices part of computing subgraphs to TensorRT to speed up the process (WIP). For specific usage, please refer to [here](http://paddlepaddle.org/documentation/docs/zh/1.1/user_guides/howto/inference/paddle_tensorrt_infer.html).


## Process of Inference Deployment

In general, the steps are:

1. Use appropriate configuration to create `PaddlePredictor`
2. Create `PaddleTensor` for input and transfer it into `PaddlePredictor` 
3. `PaddleTensor` for fetching output 

The complete process of implementing a simple model is shown below with part of details omitted.

```c++
#include "paddle_inference_api.h"

// create a config and modify associated options
paddle::NativeConfig config;
config.model_dir = "xxx";
config.use_gpu = false;
// create a native PaddlePredictor
auto predictor =
      paddle::CreatePaddlePredictor<paddle::NativeConfig>(config);
// create input tensor
int64_t data[4] = {1, 2, 3, 4};
paddle::PaddleTensor tensor;
tensor.shape = std::vector<int>({4, 1});
tensor.data.Reset(data, sizeof(data));
tensor.dtype = paddle::PaddleDType::INT64;
// create output tensor whose memory is reusable
std::vector<paddle::PaddleTensor> outputs;
// run inference
CHECK(predictor->Run(slots, &outputs));
// fetch outputs ...
```

At compile time, it is proper to co-build with `libpaddle_fluid.a/.so` . 



## Adavanced Usage

### memory management of input and output
 `data` field of `PaddleTensor` is a `PaddleBuf`, used to manage a section of memory for copying data.

There are two modes in term of memory management in `PaddleBuf` :

1. Automatic allocation and manage memory
    
    ```c++
    int some_size = 1024;
    PaddleTensor tensor;
    tensor.data.Resize(some_size);
    ```

2. Transfer outer memory

    ```c++
    int some_size = 1024;
    // You can allocate outside memory and keep it available during the usage of PaddleTensor
    void* memory = new char[some_size]; 
    
    tensor.data.Reset(memory, some_size);
    // ...
    
    // You need to release memory manually to avoid memory leak
    
    delete[] memory;
    ```

In the two modes, the first is more convenient while the second strictly controls memory management to facilitate integration with `tcmalloc` and other libraries.
 
### Upgrade performance based on contrib::AnalysisConfig

AnalyisConfig is at the stage of pre-release and protected by `namespace contrib` , which may be adjusted in the future.

Similar to `NativeConfig` , `AnalysisConfig` can create a inference engine with high performance after a series of optimization, including analysis and optimization of computing graph as well as integration and revise for some important Ops, which **largely promotes the peformance of models, such as While, LSTM, GRU** .

The usage of `AnalysisConfig` is similiar with that of `NativeConfig` but the former *only supports CPU at present and is supporting GPU more and more*.

```c++
AnalysisConfig config;
config.SetModel(dirname);                // set the directory of the model
config.EnableUseGpu(100, 0 /*gpu id*/);  // use GPU,or
config.DisableGpu();                     // use CPU
config.SwitchSpecifyInputNames(true);    // need to appoint the name of your input
config.SwitchIrOptim();     // turn on the optimization switch,and a sequence of optimizations will be executed in operation                      
```

Note that input PaddleTensor needs to be allocated. Previous examples need to be revised as follows:

```c++
auto predictor =
      paddle::CreatePaddlePredictor<paddle::contrib::AnalysisConfig>(config); // it needs AnalysisConfig here
// create input tensor
int64_t data[4] = {1, 2, 3, 4};
paddle::PaddleTensor tensor;
tensor.shape = std::vector<int>({4, 1});
tensor.data.Reset(data, sizeof(data));
tensor.dtype = paddle::PaddleDType::INT64;
tensor.name = "input0"; // name need to be set here
```

### Suggestion for Performance

1. If the CPU type permits, it's best to use the versions with support for AVX and MKL.
2. Reuse input and output `PaddleTensor` to avoid frequent memory allocation resulting in low performance
3. Try to replace `NativeConfig` with `AnalysisConfig` to perform optimization for CPU or GPU inference 

## Code Demo

[inference demos](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/inference/api/demo_ci)
