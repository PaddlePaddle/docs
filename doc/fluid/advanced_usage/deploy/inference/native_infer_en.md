# Introduction of C++ Inference API

To make deploy of inference more convenient, a series of high-level APIs provided in Fluid are used to hide different optimization of low-level APIs.

Inference library contains:

- Head file `paddle_inference_api.h` which defines all APIs定义了所有的接口
- library file `libpaddle_fluid.so` or `libpaddle_fluid.a`

Details are as follows:

## PaddleTensor

PaddleTensor defines basic format of input and output data for inference.Common fields are as follows:

- `name` is used to indicate the name of variable in model correspondent with input data.
- `shape` represents shape of a Tensor.
- `data`  is stored in `PaddleBuf` in mode of continuous storage. `PaddleBuf` can receieve outer data or independently `malloc` memory. You can refer to associated definitions in head file.
- `dtype` represents data type of Tensor.

## Use Config to create different search engines

The fundemental levels of high-level API contains many optimizations which are called engines. Transferring different Config implements overloading of different engines during the switch among them.

- `NativeConfig` native engine,consisting of native forward operators of paddle, can originally support all models trained with paddle.

- `AnalysisConfig` TensorRT mixed engine used to speed up GPU supports [TensorRT] with subgraph, supports all paddle models and aumatically segments computing subgraph to TensorRT to speed up (WIP). About specific usage,please refer to [here](http://paddlepaddle.org/documentation/docs/zh/1.1/user_guides/howto/inference/paddle_tensorrt_infer.html).


## Inference process of deploy

Steps are as follows in general:

1. Use appropriate configuration to create `PaddlePredictor`
2. Create `PaddleTensor` for input and transfer it into `PaddlePredictor` 
3. Get `PaddleTensor` of output and fetch it out

Then the complete process of implementing a simple model is shown below with part of detailed code elided.

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

Is is enough to bind to build `libpaddle_fluid.a/.so` . 



## Adavanced Usage

### memory management of input and output
field `data` of `PaddleTensor` is a `PaddleBuf`, used to manage a section of memory for copying data.

There are two modes in term of memory management in `PaddleBuf` :

1. Automatically malloc and manage memory
    
    ```c++
    int some_size = 1024;
    PaddleTensor tensor;
    tensor.data.Resize(some_size);
    ```

2. Transfer outer memory
    ```c++
    int some_size = 1024;
    // You can malloc memory outside and keep it available in the usage of PaddleTensor
    void* memory = new char[some_size]; 
    
    tensor.data.Reset(memory, some_size);
    // ...
    
    // You need to delete memory to avoid memory leak
    
    delete[] memory;
    ```

In the two modes, the first is more convenient while the second strictly controls memory management to integrate with `tcmalloc` and other libraries.

### Upgrade performance based on contrib::AnalysisConfig (pre-deploy)
*AnalyisConfig is at the stage of pre-deploy protected by `namespace contrib` ,which may be adjusted later*.

`NativeConfig`,`AnalysisConfig` can create a inference engine with high performance after a series of optimizations, including analysis and optimization of computing graph as well as integration and revise for some important Ops, which **largely upgrades the peformance of models, such as While, LSTM, GRU and so on** .

The usage of `AnalysisConfig` is similiar with that of `NativeConfig` but the former *only supports CPU at present and is supporting GPU more and more*.

```c++
AnalysisConfig config;
config.model_dir = xxx;
config.use_gpu = false;  // GPU optimization is not supported at present
config.specify_input_name = true; // it needs to set name of input
```

Attentions need to be paid that input PaddleTensor needs to be allocated.Previous examples need to be revised as follows:

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
1. Try to use CPU with AVX and MKL version if it is permitted
2. Reuse input and output `PaddleTensor` to avoid frequent malloc resulting in low performance
3. Try to replace `NativeConfig` with `AnalysisConfig` to perform optimization for CPU inference 

## Specific Code

[inference demos](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/inference/api/demo_ci)
