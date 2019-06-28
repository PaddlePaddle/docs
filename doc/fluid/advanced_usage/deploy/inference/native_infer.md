# C++ 预测 API介绍

为了更简单方便的预测部署，PaddlePaddle 提供了一套高层 API 预测接口。

预测库包含:

- 头文件主要包括： 
	- `paddle_analysis_config.h `
	- `paddle_api.h `
	- `paddle_inference_api.h`
- 库文件：
	- `libpaddle_fluid.so` 
	- `libpaddle_fluid.a`

下面是详细介绍。


## 内容
- [NativePredictor使用](#NativePredictor使用)
- [AnalysisPredictor使用](#AnalysisPredictor使用)
- [输入输出的管理](#输入输出的管理)	
- [多线程预测](#多线程预测)
- [性能建议](#性能建议)

## <a name="NativePredictor使用">NativePredictor使用</a>

`NativePredictor`为原生预测引擎，底层由 PaddlePaddle 原生的 forward operator
  组成，可以天然**支持所有Paddle 训练出的模型**。
  
#### NativePredictor 使用样例
```c++ 
#include "paddle_inference_api.h"

namespace paddle {
// 配置NativeConfig
void CreateConfig(NativeConfig *config, const std::string& model_dirname) {
  config->use_gpu=true;
  config->device=0;
  config->fraction_of_gpu_memory=0.1;
  
  /* for cpu
  config->use_gpu=false;
  config->SetCpuMathLibraryNumThreads(1);
  */
  
  // 设置模型的参数路径
  config->prog_file = model_dirname + "model";
  config->param_file = model_dirname + "params";
  // 当模型输入是多个的时候，这个配置是必要的。
  config->specify_input_name = true;
}

void RunNative(int batch_size, const std::string& model_dirname) {
  // 1. 创建NativeConfig
  NativeConfig config;
  CreateConfig(&config, model_dirname);
  
  // 2. 根据config 创建predictor
  auto predictor = CreatePaddlePredictor(config);
  
  int channels = 3;
  int height = 224;
  int width = 224;
  float *data = new float[batch_size * channels * height * width];

  // 3. 创建输入 tensor 
  PaddleTensor tensor;
  tensor.name = "image";
  tensor.shape = std::vector<int>({batch_size, channels, height, width});
  tensor.data = PaddleBuf(static_cast<void *>(data),
                          sizeof(float) * (batch_size * channels * height * width));
  tensor.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

  // 4. 创建输出 tensor
  std::vector<PaddleTensor> outputs;
  // 5. 预测
  predictor->Run(paddle_tensor_feeds, &outputs, batch_size);

  const size_t num_elements = outputs.front().data.length() / sizeof(float);
  auto *data_out = static_cast<float *>(outputs.front().data.data());
}
}  // namespace paddle

int main() { 
  // 模型下载地址 http://paddle-inference-dist.cdn.bcebos.com/tensorrt_test/mobilenet.tar.gz
  paddle::RunNative(1, "./mobilenet");
  return 0;
}
```

## <a name="AnalysisPredictor使用"> AnalysisPredictor使用</a>
AnalysisConfig 创建了一个高性能预测引擎。该引擎通过对计算图的分析，完成对计算图的一系列的优化（Op 的融合, MKLDNN，TRT等底层加速库的支持 etc），大大提升预测引擎的性能。 

#### AnalysisPredictor 使用样例

```c++
#include "paddle_inference_api.h"

namespace paddle {
void CreateConfig(AnalysisConfig* config, const std::string& model_dirname) {
  // 模型从磁盘进行加载
  config->SetModel(model_dirname + "/model",                                                                                             
                      model_dirname + "/params");  
  // config->SetModel(model_dirname);
  // 如果模型从内存中加载，可以使用SetModelBuffer接口
  // config->SetModelBuffer(prog_buffer, prog_size, params_buffer, params_size); 
  config->EnableUseGpu(10 /*the initial size of the GPU memory pool in MB*/,  0 /*gpu_id*/);
  
  /* for cpu 
  config->DisableGpu();
  config->EnableMKLDNN();   // 可选
  config->SetCpuMathLibraryNumThreads(10);
  */
 
  // 当使用ZeroCopyTensor的时候，此处一定要设置为false。
  config->SwitchUseFeedFetchOps(false);
  // 当多输入的时候，此处一定要设置为true
  config->SwitchSpecifyInputNames(true);
  config->SwitchIrDebug(true); // 开关打开，会在每个图优化过程后生成dot文件，方便可视化。
  // config->SwitchIrOptim(false); // 默认为true。如果设置为false，关闭所有优化，执行过程同 NativePredictor
  // config->EnableMemoryOptim(); // 开启内存/显存复用
}

void RunAnalysis(int batch_size, std::string model_dirname) {
  // 1. 创建AnalysisConfig
  AnalysisConfig config;
  CreateConfig(&config, model_dirname);
  
  // 2. 根据config 创建predictor
  auto predictor = CreatePaddlePredictor(config);
  int channels = 3;
  int height = 224;
  int width = 224;
  float input[batch_size * channels * height * width] = {0};
  
  // 3. 创建输入
  // 同NativePredictor样例一样，此处可以使用PaddleTensor来创建输入
  // 以下的代码中使用了ZeroCopy的接口，同使用PaddleTensor不同的是：此接口可以避免预测中多余的cpu copy，提升预测性能。
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu(input);

  // 4. 运行
  CHECK(predictor->ZeroCopyRun());
   
  // 5. 获取输出
  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());
}
}  // namespace paddle

int main() { 
  // 模型下载地址 http://paddle-inference-dist.cdn.bcebos.com/tensorrt_test/mobilenet.tar.gz
  paddle::RunAnalysis(1, "./mobilenet");
  return 0;
}


```

## <a name="输入输出的管理"> 输入输出的管理</a>
### PaddleTensor 的使用
PaddleTensor可用于NativePredictor和AnalysisPredictor，在 NativePredictor样例中展示了PaddleTensor的使用方式。
PaddleTensor 定义了预测最基本的输入输出的数据格式，常用字段如下：

- `name`，类型：string，用于指定输入数据对应的模型中variable的名字
- `shape`，类型：`vector<int>`, 表示一个Tensor的shape
- `data`，类型：`PaddleBuf`, 数据以连续内存的方式存储在`PaddleBuf`中，`PaddleBuf`可以接收外面的数据或者独立`malloc`内存，详细可以参考头文件中相关定义。
- `dtype`，类型：`PaddleType`, 有`PaddleDtype::FLOAT32`, `PaddleDtype::INT64`, `PaddleDtype::INT32`三种, 表示 Tensor 的数据类型。
- `lod`，类型：`vector<vector<size_t>>`，在处理变长输入的时候，需要对 `PaddleTensor`设置LoD信息。可以参考[LoD-Tensor使用说明](../../../user_guides/howto/basic_concept/lod_tensor.html)


### ZeroCopyTensor的使用
ZeroCopyTensor的使用可避免预测时候准备输入以及获取输出时多余的数据copy，提高预测性能。**只可用于AnalysisPredictor**。    

**Note:**使用ZeroCopyTensor，务必在创建config时设置`config->SwitchUseFeedFetchOps(false)`

```
// 通过创建的AnalysisPredictor获取输入和输出的tensor
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputTensor(input_names[0]);
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputTensor(output_names[0]);

// 对tensor进行reshape
input_t->Reshape({batch_size, channels, height, width});

// 通过copy_from_cpu接口，将cpu数据输入；通过copy_to_cpu接口，将输出数据copy到cpu
input_t->copy_from_cpu<float>(input_data /*数据指针*/);
output_t->copy_to_cpu(out_data /*数据指针*/);

// 设置LOD 
std::vector<std::vector<size_t>> lod_data = {{0}, {0}};
input_t->SetLoD(lod_data);

// 获取tensor数据指针
float *input_d = input_t->mutable_data<float>(PaddlePlace::kGPU);  // CPU下使用PaddlePlace::kCPU
int output_size;
float *output_d = output_t->data<float>(PaddlePlace::kGPU, &output_size);
```

## <a name="多线程预测"> 多线程预测</a>


多线程场景下，每个服务线程执行同一种模型，支持 CPU 和 GPU。

下面演示最简单的实现，用户需要根据具体应用场景做相应的调整

```c++
auto main_predictor = paddle::CreatePaddlePredictor(config);

const int num_threads = 10;  // 假设有 10 个服务线程
std::vector<std::thread> threads;
std::vector<decl_type(main_predictor)> predictors;

// 线程外创建所有的predictor
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

// 线程join
for (auto& t : threads) {
    if (t.joinable()) t.join();
}

// 结束
```


## <a name="性能建议"> 性能建议</a>

1. 在CPU型号允许的情况下，尽量使用带AVX和MKL的版本
2. CPU或GPU预测，可以尝试把`NativeConfig`改成`AnalysisConfig`来进行优化
3. 尽量使用`ZeroCopyTensor`避免过多的内存copy
4. CPU下可以尝试使用Intel的`MKLDNN`加速
5. GPU 下可以尝试打开`TensorRT`子图加速引擎, 通过计算图分析，Paddle可以自动将计算图中部分子图切割，并调用NVidia的 `TensorRT` 来进行加速。
详细内容可以参考 [Paddle-TRT 子图引擎](./paddle_tensorrt_infer.html)
