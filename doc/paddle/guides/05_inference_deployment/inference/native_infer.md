# C++ 预测 API介绍

为了更简单方便地预测部署，PaddlePaddle 提供了一套高层 C++ API 预测接口。下面是详细介绍。

如果您在使用2.0之前的Paddle，请参考[旧版API](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/advanced_guide/inference_deployment/inference/native_infer.html)文档，升级到新版API请参考[推理升级指南](#推理升级指南)。

## 内容

- [使用Predictor进行高性能预测](#使用Predictor进行高性能预测)
- [使用Config管理预测配置](#使用Config管理预测配置)
- [使用Tensor管理输入/输出](#使用Tensor管理输入/输出)
- [使用PredictorPool在多线程下进行预测](#使用PredictorPool在多线程下进行预测)
- [C++预测样例编译测试](#C++预测样例编译测试)
- [性能调优](#性能调优)
- [推理升级指南](#推理升级指南)
- [C++ API](#C++_API)


## <a name="使用Predictor进行高性能预测"> 使用Predictor进行高性能预测</a>

Paddle Inference采用 Predictor 进行预测。Predictor 是一个高性能预测引擎，该引擎通过对计算图的分析，完成对计算图的一系列的优化（如OP的融合、内存/显存的优化、 MKLDNN，TensorRT 等底层加速库的支持等），能够大大提升预测性能。

为了展示完整的预测流程，下面是一个使用 Predictor 进行预测的完整示例，其中涉及到的具体概念和配置会在后续部分展开详细介绍。

#### Predictor 预测示例

``` c++
#include "paddle_inference_api.h"

namespace paddle_infer {
void CreateConfig(Config* config, const std::string& model_dirname) {
  // 模型从磁盘进行加载
  config->SetModel(model_dirname + "/model",
                   model_dirname + "/params");
  // config->SetModel(model_dirname);
  // 如果模型从内存中加载，可以使用SetModelBuffer接口
  // config->SetModelBuffer(prog_buffer, prog_size, params_buffer, params_size);
  config->EnableUseGpu(100 /*设定GPU初始显存池为MB*/,  0 /*设定GPU ID为0*/); //开启GPU预测

  /* for cpu
  config->DisableGpu();
  config->EnableMKLDNN();   // 开启MKLDNN加速
  config->SetCpuMathLibraryNumThreads(10);
  */

  config->SwitchIrDebug(true);         // 可视化调试选项，若开启，则会在每个图优化过程后生成dot文件
  // config->SwitchIrOptim(false);     // 默认为true。如果设置为false，关闭所有优化
  // config->EnableMemoryOptim();     // 开启内存/显存复用
}

void RunAnalysis(int batch_size, std::string model_dirname) {
  // 1. 创建Config
  Config config;
  CreateConfig(&config, model_dirname);

  // 2. 根据config 创建predictor，并准备输入数据，此处以全0数据为例
  auto predictor = CreatePredictor(config);
  int channels = 3;
  int height = 224;
  int width = 224;
  float input[batch_size * channels * height * width] = {0};

  // 3. 创建输入
  // 使用了ZeroCopy接口，可以避免预测中多余的CPU copy，提升预测性能
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(input);

  // 4. 运行预测引擎
  CHECK(predictor->Run());

  // 5. 获取输出
  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());
}
}  // namespace paddle_infer

int main() {
  // 模型下载地址 http://paddle-inference-dist.cdn.bcebos.com/tensorrt_test/mobilenet.tar.gz
  paddle_infer::RunAnalysis(1, "./mobilenet");
  return 0;
}

```

## <a name="使用Config管理预测配置"> 使用Config管理预测配置</a>

Config管理Predictor的预测配置，提供了模型路径设置、预测引擎运行设备选择以及多种优化预测流程的选项。配置方法如下：

#### 通用优化配置
``` c++
config->SwitchIrOptim(true);  // 开启计算图分析优化，包括OP融合等
config->EnableMemoryOptim();  // 开启内存/显存复用
```

#### 设置模型和参数路径
从磁盘加载模型时，根据模型和参数文件存储方式不同，设置Config加载模型和参数的路径有两种形式：

* 非combined形式：模型文件夹`model_dir`下存在一个模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为`__model__`。
``` c++
config->SetModel("./model_dir");
```

* combined形式：模型文件夹`model_dir`下只有一个模型文件`model`和一个参数文件`params`时，传入模型文件和参数文件路径。
``` c++
config->SetModel("./model_dir/model", "./model_dir/params");
```


#### 配置CPU预测

``` c++
config->DisableGpu();          // 禁用GPU
config->EnableMKLDNN();            // 开启MKLDNN，可加速CPU预测
config->SetCpuMathLibraryNumThreads(10);        // 设置CPU Math库线程数，CPU核心数支持情况下可加速预测
```

**note**

如果在输入shape为变长时开启MKLDNN加速预测，需要通过`SetMkldnnCacheCapacity`接口设置MKLDNN缓存的不同输入shape的数目，否则可能会出现内存泄漏。使用方法如下：
```c++
config->SetMkldnnCacheCapacity(100); // 缓存100个不同的输入shape
```

#### 配置GPU预测
``` c++
config->EnableUseGpu(100, 0); // 初始化100M显存，使用GPU ID为0
config->GpuDeviceId();        // 返回正在使用的GPU ID
// 开启TensorRT预测，可提升GPU预测性能，需要使用带TensorRT的预测库
config->EnableTensorRtEngine(1 << 20             /*workspace_size*/,
                             batch_size        /*max_batch_size*/,
                             3                 /*min_subgraph_size*/,
                             PrecisionType::kFloat32 /*precision*/,
                             false             /*use_static*/,
                             false             /*use_calib_mode*/);
```


## <a name="使用Tensor管理输入/输出"> 使用Tensor管理输入/输出</a>

Tensor是Predictor的输入/输出数据结构。

``` c++
// 通过创建的Predictor获取输入和输出的tensor
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputHandle(input_names[0]);
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputHandle(output_names[0]);

// 对tensor进行reshape
input_t->Reshape({batch_size, channels, height, width});

// 通过CopyFromCpu接口，将cpu数据输入；通过CopyToCpu接口，将输出数据copy到cpu
input_t->CopyFromCpu<float>(input_data /*数据指针*/);
output_t->CopyToCpu(out_data /*数据指针*/);

// 设置LOD
std::vector<std::vector<size_t>> lod_data = {{0}, {0}};
input_t->SetLoD(lod_data);

// 获取Tensor数据指针
float *input_d = input_t->mutable_data<float>(PaddlePlace::kGPU);  // CPU下使用PaddlePlace::kCPU
int output_size;
float *output_d = output_t->data<float>(PaddlePlace::kGPU, &output_size);
```

## <a name="使用PredictorPool在多线程下进行预测"> 使用PredictorPool在多线程下进行预测</a>

`PredictorPool`对`Predictor`进行管理。`PredictorPool`对`Predictor`进行了简单的封装，通过传入config和thread的数目来完成初始化，在每个线程中，根据自己的线程id直接从池中取出对应的`Predictor`来完成预测过程。

```c++
# 服务初始化时，完成PredictorPool的初始化
PredictorPool pool(config, thread_num);

# 根据线程id来获取Predictor
auto predictor = pool.Retrive(thread_id);

# 使用Predictor进行预测
...
```

## <a name="C++预测样例编译测试"> C++预测样例编译测试</a>

1. 下载或编译paddle预测库，参考[安装与编译C++预测库](./build_and_install_lib_cn.html)。
2. 下载[预测样例](https://paddle-inference-dist.bj.bcebos.com/samples/sample.tgz)并解压，进入`sample/inference`目录下。  

    `inference` 文件夹目录结构如下：

    ``` shell
    inference
    ├── CMakeLists.txt
    ├── mobilenet_test.cc
    ├── thread_mobilenet_test.cc
    ├── mobilenetv1
    │   ├── model
    │   └── params
    ├── run.sh
    └── run_impl.sh
    ```

    - `mobilenet_test.cc` 为单线程预测的C++源文件
    - `thread_mobilenet_test.cc` 为多线程预测的C++源文件  
    - `mobilenetv1` 为模型文件夹
    - `run.sh` 为预测运行脚本文件

3. 配置编译与运行脚本

    编译运行预测样例之前，需要根据运行环境配置编译与运行脚本`run.sh`。`run.sh`的选项与路径配置的部分如下：

    ``` shell
    # 设置是否开启MKL、GPU、TensorRT，如果要使用TensorRT，必须打开GPU
    WITH_MKL=ON
    WITH_GPU=OFF
    USE_TENSORRT=OFF

    # 按照运行环境设置预测库路径、CUDA库路径、CUDNN库路径、TensorRT路径、模型路径
    LIB_DIR=YOUR_LIB_DIR
    CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
    CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
    TENSORRT_ROOT_DIR=YOUR_TENSORRT_ROOT_DIR
    MODEL_DIR=YOUR_MODEL_DIR
    ```

    按照实际运行环境配置`run.sh`中的选项开关和所需lib路径。

4. 编译与运行样例  

    ``` shell
    sh run.sh
    ```

## <a name="性能调优"> 性能调优</a>

### CPU下预测
1. 在CPU型号允许的情况下，尽量使用带AVX和MKL的版本。
2. 可以尝试使用Intel的 MKLDNN 加速。
3. 在CPU可用核心数足够时，可以将设置`config->SetCpuMathLibraryNumThreads(num);`中的num值调高一些。

### GPU下预测
1. 可以尝试打开 TensorRT 子图加速引擎, 通过计算图分析，Paddle可以自动将计算图中部分子图融合，并调用NVIDIA的 TensorRT 来进行加速，详细内容可以参考 [使用Paddle-TensorRT库预测](../../performance_improving/inference_improving/paddle_tensorrt_infer.html)。

### 多线程预测
Paddle Inference支持通过在不同线程运行多个Predictor的方式来优化预测性能，支持CPU和GPU环境。

使用多线程预测的样例详见[C++预测样例编译测试](#C++预测样例编译测试)中下载的[预测样例](https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/paddle_inference_sample_v1.7.tar.gz)中的
`thread_mobilenet_test.cc`文件。可以将`run.sh`中`mobilenet_test`替换成`thread_mobilenet_test`再执行

```
sh run.sh
```

即可运行多线程预测样例。

## <a name="推理升级指南"> 推理升级指南</a>

2.0对API做了整理，简化了写法，以及去掉了历史上冗余的概念。

新的 API 为纯增，原有 API 保持不变，在后续版本会逐步删除。

重要变化：

- 命名空间从 `paddle` 变更为 `paddle_infer`
- `PaddleTensor`, `PaddleBuf` 等被废弃，`ZeroCopyTensor` 变为默认 Tensor 类型，并更名为 `Tensor`
- 新增 `PredictorPool` 工具类简化多线程 predictor 的创建，后续也会增加更多周边工具
- `CreatePredictor` (原 `CreatePaddlePredictor`) 的返回值由 `unique_ptr` 变为 `shared_ptr` 以避免 Clone 后析构顺序出错的问题

API 变更

| 原有命名                     | 现有命名                     | 行为变化                      |
| ---------------------------- | ---------------------------- | ----------------------------- |
| 头文件 `paddle_infer.h`      | 无变化                       | 包含旧接口，保持向后兼容      |
| 无                           | `paddle_inference_api.h`     | 新API，可以与旧接口并存       |
| `CreatePaddlePredictor`      | `CreatePredictor`            | 返回值变为 shared_ptr         |
| `ZeroCopyTensor`             | `Tensor`                     | 无                            |
| `AnalysisConfig`             | `Config`                     | 无                            |
| `TensorRTConfig`             | 废弃                         |                               |
| `PaddleTensor` + `PaddleBuf` | 废弃                         |                               |
| `Predictor::GetInputTensor`  | `Predictor::GetInputHandle`  | 无                            |
| `Predictor::GetOutputTensor` | `Predictor::GetOutputHandle` | 无                            |
|                              | `PredictorPool`              | 简化创建多个 predictor 的支持 |

使用新 C++ API 的流程与之前完全一致，只有命名变化

```c++
#include "paddle_infernce_api.h"
using namespace paddle_infer;

Config config;
config.SetModel("xxx_model_dir");

auto predictor = CreatePredictor(config);

// Get the handles for the inputs and outputs of the model
auto input0 = predictor->GetInputHandle("X");
auto output0 = predictor->GetOutputHandle("Out");

for (...) {
    // Assign data to input0
  MyServiceSetData(input0);

  predictor->Run();

  // get data from the output0 handle
  MyServiceGetData(output0);
}
```

## <a name="C++_API"> C++ API</a>

##### CreatePredictor

```c++
std::shared_ptr<Predictor> CreatePredictor(const Config& config);
```

`CreatePredictor`用来根据`Config`构建预测引擎。

示例：

```c++
// 设置Config
Config config;
config.SetModel(FLAGS_model_dir);

// 根据Config创建Predictor
std::shared_ptr<Predictor> predictor = CreatePredictor(config);
```

参数：

- `config(Config)` - 用于构建Predictor的配置信息

返回：`Predictor`智能指针

返回类型：`std::shared_ptr<Predictor>`

##### GetVersion()

```c++
std::string GetVersion();
```

打印Paddle Inference的版本信息。

参数：

- `None`

返回：版本信息

返回类型：`std::string`

##### PlaceType

```c++
enum class PaddlePlace { kUNK };
using PlaceType = paddle::PaddlePlace;
```

PlaceType为目标设备硬件类型，用户可以根据应用场景选择硬件平台类型。

枚举变量`PlaceType`的所有可能取值包括：

`{kUNK, kCPU, kGPU}`

##### PrecisionType

```c++
enum class Precision { kFloat32 };
using PrecisionType = paddle::AnalysisConfig::Precision;
```

`PrecisionType`设置模型的运行精度，默认值为kFloat32(float32)。

枚举变量`PrecisionType`的所有可能取值包括：

`{kFloat32, kInt8, kHalf}`

##### DataType

```c++
enum class PaddleDType { FLOAT32 };
using DataType = paddle::PaddleDType;
```

`DataType`为模型中Tensor的数据精度，默认值为FLOAT32(float32)。

枚举变量`DataType`的所有可能取值包括：

`{FLOAT32, INT64, INT32, UINT8}`

##### GetNumBytesOfDataType

```c++
int GetNumBytesOfDataType(DataType dtype);
```

获取各个`DataType`对应的字节数。

参数：

- `dtype` - DataType枚举

返回：字节数

返回类型：`int`


##### Predictor

```c++
class Predictor;
```

`Predictor`是Paddle Inference的预测器，由`CreatePredictor`根据`Config`进行创建。用户可以根据Predictor提供的接口设置输入数据、执行模型预测、获取输出等.

示例：

```c++
using namespace paddle_infer;

Config config;
config.SetModel("xxx_model_dir");

auto predictor = CreatePredictor(config);

// Get the handles for the inputs and outputs of the model
auto input0 = predictor->GetInputHandle("X");
auto output0 = predictor->GetOutputHandle("Out");

for (...) {
    // Assign data to input0
  MyServiceSetData(input0);

  predictor->Run();

  // get data from the output0 handle
  MyServiceGetData(output0);
}
```

###### GetInputNames()

获取所有输入Tensor的名称。

参数：

- `None`

返回：所有输入Tensor的名称

返回类型：`std::vector<std::string>`

###### GetOutputNames()

获取所有输出Tensor的名称。

参数：

- `None`

返回：所有输出Tensor的名称

返回类型：`std::vector<std::string>`

###### GetInputHandle(const std::string& name)

根据名称获取输入Tensor的句柄。

参数：

- `name` - Tensor的名称

返回：指向`Tensor`的指针

返回类型：`std::unique_ptr<Tensor>`

###### GetOutputHandle(const std::string& name)

根据名称获取输出Tensor的句柄。

参数：

- `name` - Tensor的名称

返回：指向`Tensor`的指针

返回类型：`std::unique_ptr<Tensor>`

###### Run()

执行模型预测，需要在***设置输入数据后***调用。

参数：

- `None`

返回：`None`

返回类型：`void`

###### ClearIntermediateTensor()

释放临时tensor，将其所占空间归还显/内存池。

参数：

- `None`

返回：`None`

返回类型：`void`

###### TryShrinkMemory()

释放临时tensor，并检查显/内存池中是否有可以释放的chunk，若有则释放chunk，降低显/内存占用（显/内存池可认为是`list<chunk>`组成，如果chunk空闲，则可通过释放chunk来降低显/内存占用），demo示例可参考[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/test/shrink_memory)。

参数：

- `None`

返回：`None`

返回类型：`void`

###### Clone()

根据该Predictor，克隆一个新的Predictor，两个Predictor之间共享权重。

参数：

- `None`

返回：新的Predictor

返回类型：`std::unique_ptr<Predictor>`


##### Tensor

```c++
class Tensor;
```

Tensor是Paddle Inference的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置Shape、数据、LoD信息等。

*注意：用户应使用`Predictor`的`GetInputHandle`和`GetOuputHandle`接口获取输入/输出的`Tensor`。*

示例：

```c++
// 通过创建的Predictor获取输入和输出的tensor
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputHandle(input_names[0]);
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputHandle(output_names[0]);

// 对tensor进行reshape
input_t->Reshape({batch_size, channels, height, width});

// 通过CopyFromCpu接口，将cpu数据输入；通过CopyToCpu接口，将输出数据copy到cpu
input_t->CopyFromCpu<float>(input_data /*数据指针*/);
output_t->CopyToCpu(out_data /*数据指针*/);

// 设置LOD
std::vector<std::vector<size_t>> lod_data = {{0}, {0}};
input_t->SetLoD(lod_data);

// 获取Tensor数据指针
float *input_d = input_t->mutable_data<float>(PlaceType::kGPU);  // CPU下使用PlaceType::kCPU
int output_size;
float *output_d = output_t->data<float>(PlaceType::kGPU, &output_size);
```

###### Reshape(shape)

设置Tensor的维度信息。

参数：

- `shape(const std::vector<int>&)` - 维度信息

返回：`None`

返回类型：`void`

###### shape()

获取Tensor的维度信息。

参数：

- `None`

返回：Tensor的维度信息

返回类型：`std::vector<int>`

###### CopyFromCpu(data)

```c++
template <typename T>
void CopyFromCpu(const T* data);
```

从cpu获取数据，设置到tensor内部。

示例：

```c++
// float* data = ...;
auto in_tensor = predictor->GetInputHandle("in_name");
in_tensor->CopyFromCpu(data);
```

参数：

- `data(const T*)` - cpu数据指针

返回：`None`

返回类型：`void`

###### CopyToCpu(data)

```c++
template <typename T>
void CopyToCpu(T* data);
```

示例：

```c++
std::vector<float> data(100);
auto out_tensor = predictor->GetOutputHandle("out_name");
out_tensor->CopyToCpu(data.data());
```

参数：

- `data(T*)` - cpu数据指针

返回：`None`

返回类型：`void`


###### data<T>(place, size)

```c++
template <typename T>
T* data(PlaceType* place, int* size) const;
```

获取Tensor的底层数据的常量指针，用于读取Tensor数据。

示例：

```c++
PlaceType place;
int size;
auto out_tensor = predictor->GetOutputHandle("out_name");
float* data = out_tensor->data<float>(&place, &size);
```

参数：

- `place(PlaceType*)` - 获取tensor的PlaceType
- `size(int*)` - 获取tensor的size

返回：数据指针

返回类型：`T*`

###### mutable_data<T>(place)

```c++
template <typename T>
T* mutable_data(PlaceType place);
```

获取Tensor的底层数据的指针，用于设置Tensor数据。

```c++
auto in_tensor = predictor->GetInputHandle("in_name");
float* data = out_tensor->mutable_data<float>(PlaceType::kCPU);
data[0] = 1.;
```

参数：

- `place(PlaceType)` - 设备信息

返回：`Tensor`底层数据指针

返回类型：`T*`

###### SetLoD(lod)

设置Tensor的LoD信息。

参数：

- `lod(const std::vector<std::vector<size_t>>)` - Tensor的LoD信息

返回：`None`

返回类型：`void`

###### lod()

获取Tensor的LoD信息

参数：

- `None`

返回：`Tensor`的LoD信息

返回类型：`std::vector<std::vector<size_t>>`

###### type()

tensor的DataType信息。

参数：

- `None`

返回：`Tensor`的DataType信息

返回类型：`DataType`

###### name()

tensor对应的name。

参数：

- `None`

返回：`Tensor`对应的name

返回类型：`std::string`

##### Config

```c++
class Config;
```

`Config`用来配置构建`Predictor`的配置信息，如模型路径、是否开启gpu等等。

示例：

```c++
Config config;
config.SetModel(FLAGS_model_dir);
config.DisableGpu();
config->SwitchIrOptim(false);     // 默认为true。如果设置为false，关闭所有优化
config->EnableMemoryOptim();     // 开启内存/显存复用
```

###### SetModel(const std::string& model_dir)

设置模型文件路径，当需要从磁盘加载非combine模式时使用。

参数：

- `model_dir` - 模型文件夹路径

返回：`None`

返回类型：`void`


###### model_dir()

获取模型文件夹路径。

参数：

- `None`

返回：模型文件夹路径

返回类型：`string`


###### SetModel(const std::string& prog, const std::string& params)

设置模型文件路径，当需要从磁盘加载combine模式时使用。

参数：

- `prog` - 模型文件路径
- `params` - 模型参数文件路径

返回：`None`

返回类型：`void`

###### SetProgFile(const std::string& prog)

设置模型文件路径。

参数：

- `prog` - 模型文件路径

返回：`None`

返回类型：`void`

###### prog_file()

获取模型文件路径。

参数：

- `None`

返回：模型文件路径

返回类型：`string`

###### SetParamsFile(const std::string& params)

设置模型参数文件路径。

参数：

- `params` - 模型文件路径

返回：`None`

返回类型：`void`

###### params_file()

获取模型参数文件路径。

参数：

- `None`

返回：模型参数文件路径

返回类型：`string`

###### SetModelBuffer(const char* prog_buffer, size_t prog_buffer_size, const char* params_buffer, size_t params_buffer_size)

从内存加载模型。

参数：

- `prog_buffer` - 内存中模型结构数据
- `prog_buffer_size` - 内存中模型结构数据的大小
- `params_buffer` - 内存中模型参数数据
- `params_buffer_size` - 内存中模型参数数据的大小

返回：`None`

返回类型：`void`

###### model_from_memory()

判断是否从内存中加载模型。

参数：

- `None`

返回：是否从内存中加载模型

返回类型：`bool`

###### SetOptimCacheDir(const std::string& opt_cache_dir)

设置缓存路径。

参数：

- `opt_cache_dir` - 缓存路径

返回：`None`

返回类型：`void`

###### DisableFCPadding（）

关闭fc padding。

参数：

- `None`

返回：`None`

返回类型：`void`

###### use_fc_padding()

判断是否启用fc padding。

参数：

- `None`

返回：是否启用fc padding

返回类型：`bool`

###### EnableUseGpu(uint64_t memory_pool_init_size_mb, int device_id = 0)

启用gpu。

参数：

- `memory_pool_init_size_mb` - 初始化分配的gpu显存，以MB为单位
- `device_id` - 设备id

返回：`None`

返回类型：`void`

###### DisableGpu()

禁用gpu。

参数：

- `None`

返回：`None`

返回类型：`void`

###### use_gpu()

是否启用gpu。

参数：

- `None`

返回：是否启用gpu

返回类型：`bool`

###### gpu_device_id()

获取gpu的device id。

参数：

- `None`

返回：gpu的device id

返回类型：`int`

###### memory_pool_init_size_mb()

获取gpu的初始显存大小。

参数：

- `None`

返回：初始的显存大小

返回类型：`int`

###### fraction_of_gpu_memory_for_pool()

初始化显存占总显存的百分比

参数：

- `None`

返回：初始的显存占总显存的百分比

返回类型：`float`

###### EnableCUDNN()

启用cudnn。

参数：

- `None`

返回：`None`

返回类型：`void`

###### cudnn_enabled()

是否启用cudnn。

参数：

- `None`

返回：是否启用cudnn

返回类型：`bool`

###### EnableXpu(int l3_workspace_size)

启用xpu。

参数：

- `l3_workspace_size` - l3 cache分配的显存大小

返回：`None`

返回类型：`void`

###### SwitchIrOptim(int x=true)

设置是否开启ir优化。

参数：

- `x` - 是否开启ir优化，默认打开

返回：`None`

返回类型：`void`

###### ir_optim()

是否开启ir优化。

参数：

- `None`

返回：是否开启ir优化

返回类型：`bool`

###### SwitchUseFeedFetchOps(int x = true)

设置是否使用feed，fetch op，仅内部使用。

参数：

- `x` - 是否使用feed, fetch op

返回：`None`

返回类型：`void`

###### use_feed_fetch_ops_enabled()

是否使用feed，fetch op。

参数：

- `None`

返回：是否使用feed，fetch op

返回类型：`bool`

###### SwitchSpecifyInputNames(bool x = true)

设置是否需要指定输入tensor的name。

参数：

- `x` - 是否指定输入tensor的name

返回：`None`

返回类型：`void`

###### specify_input_name()

是否需要指定输入tensor的name。

参数：

- `None`

返回：是否需要指定输入tensor的name

返回类型：`bool`

###### EnableTensorRtEngine(int workspace_size = 1 << 20, int max_batch_size = 1, int min_subgraph_size = 3, Precision precision = Precision::kFloat32, bool use_static = false, bool use_calib_mode = true)

设置是否启用TensorRT。

参数：

- `workspace_size` - 指定TensorRT使用的工作空间大小
- `max_batch_size` - 设置最大的batch大小，运行时batch大小不得超过此限定值
- `min_subgraph_size` - Paddle-TRT是以子图的形式运行，为了避免性能损失，当子图内部节点个数大于min_subgraph_size的时候，才会使用Paddle-TRT运行
- `precision` - 指定使用TRT的精度，支持FP32（kFloat32），FP16（kHalf），Int8（kInt8）
- `use_static` - 如果指定为true，在初次运行程序的时候会将TRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成
- `use_calib_mode` - 若要运行Paddle-TRT int8离线量化校准，需要将此选项设置为true

返回：`None`

返回类型：`void`

###### tensorrt_engine_enabled()

是否启用tensorRT。

参数：

- `None`

返回：是否启用tensorRT

返回类型：`bool`

###### SetTRTDynamicShapeInfo(std::map<std::string, std::vector<int>> min_input_shape, std::map<std::string, std::vector<int>> max_input_shape, std::map<std::string, std::vector<int>> optim_input_shape, bool disable_trt_plugin_fp16 = false)

设置tensorRT的动态shape。

参数：

- `min_input_shape` - tensorRT子图支持动态shape的最小shape
- `max_input_shape` - tensorRT子图支持动态shape的最大shape
- `optim_input_shape` - tensorRT子图支持动态shape的最优shape
- `disable_trt_plugin_fp16` - 设置tensorRT的plugin不在fp16精度下运行

返回：`None`

返回类型：`void`

###### EnableLiteEngine(AnalysisConfig::Precision precision_mode = Precsion::kFloat32, bool zero_copy = false, const std::vector<std::string>& passes_filter = {}, const std::vector<std::string>& ops_filter = {})

启用lite子图。

参数：

- `precision_mode` - lite子图的运行精度
- `zero_copy` - 启用zero_copy，lite子图与paddle inference之间共享数据
- `passes_filter` - 设置lite子图的pass
- `ops_filter` - 设置不使用lite子图运行的op

返回：`None`

返回类型：`void`

###### lite_engine_enabled()

是否启用lite子图。

参数：

- `None`

返回：是否启用lite子图

返回类型：`bool`

###### SwitchIrDebug(int x = true)

设置是否在图分析阶段打印ir，启用后会在每一个pass后生成dot文件。

参数：

- `x` - 是否打印ir

返回：`None`

返回类型：`void`

###### EnableMKLDNN()

启用mkldnn。

参数：

- `None`

返回：`None`

返回类型：`void`

###### SetMkldnnCacheCapacity(int capacity)

设置mkldnn针对不同输入shape的cache容量大小，MKLDNN cache设计文档请参考[链接](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/mkldnn/caching/caching.md)

参数：

- `capacity` - cache容量大小

返回：`None`

返回类型：`void`

###### mkldnn_enabled()

是否启用mkldnn。

参数：

- `None`

返回：是否启用mkldnn

返回类型：`bool`

###### SetMKLDNNOp(std::unordered_set<std::string> op_list)

指定优先使用mkldnn加速的op列表。

参数：

- `op_list` - 优先使用mkldnn的op列表

返回：`None`

返回类型：`void`

###### EnableMkldnnQuantizer()

启用mkldnn量化。

参数：

- `None`

返回：`None`

返回类型：`void`

###### mkldnn_quantizer_enabled()

是否启用mkldnn量化。

参数：

- `None`

返回：是否启用mkldnn量化

返回类型：`bool`

###### EnableMkldnnBfloat16()

启用mkldnn bf16。

参数：

- `None`

返回：`None`

返回类型：`void`

###### mkldnn_bfloat16_enabled()

是否启用mkldnn bf16。

参数：

- `None`

返回：是否启用mkldnn bf16

返回类型：`bool`

###### mkldnn_quantizer_config()

返回mkldnn量化config。

参数：

- `None`

返回：mkldnn量化config

返回类型：`MkldnnQuantizerConfig`

###### SetCpuMathLibraryNumThreads(int cpu_math_library_num_threads)

设置cpu blas库计算线程数。

参数：

- `cpu_math_library_num_threads` - blas库计算线程数

返回：`None`

返回类型：`void`

###### cpu_math_library_num_threads()

cpu blas库计算线程数。

参数：

- `None`

返回：cpu blas库计算线程数。

返回类型：`int`

###### ToNativeConfig()

转化为NativeConfig，不推荐使用。

参数：

- `None`

返回：当前Config对应的NativeConfig

返回类型：`NativeConfig`

###### EnableGpuMultiStream()

开启线程流，目前的行为是为每一个线程绑定一个流，在将来该行为可能改变。

参数：

- `None`

返回：`None`

返回类型：`void`

###### thread_local_stream_enabled()

是否启用线程流。

参数：

- `None`

返回：是否启用线程流。

返回类型：`bool`

###### EnableMemoryOptim()

开启内/显存复用，具体降低内存效果取决于模型结构。

参数：

- `None`

返回：`None`

返回类型：`void`

###### enable_memory_optim()

是否开启内/显存复用。

参数：

- `None`

返回：是否开启内/显存复用。

返回类型：`bool`

###### EnableProfile()

打开profile，运行结束后会打印所有op的耗时占比。

参数：

- `None`

返回：`None`

返回类型：`void`

###### profile_enabled()

是否开启profile。

参数：

- `None`

返回：是否开启profile

返回类型：`bool`

###### DisableGlogInfo()

去除Paddle Inference运行中的log。

参数：

- `None`

返回：`None`

返回类型：`void`

###### glog_info_disabled()

是否禁用了log。

参数：

- `None`

返回：是否禁用了log

返回类型：`bool`

###### SetInValid()

设置Config为无效状态，仅内部使用，保证每一个Config仅用来初始化一次Predictor。

参数：

- `None`

返回：`None`

返回类型：`void`

###### is_valid()

当前Config是否有效。

参数：

- `None`

返回：Config是否有效

返回类型：`bool`

###### pass_builder()

返回pass_builder，用来自定义图分析阶段选择的ir。

示例:

```c++
Config config;
auto pass_builder = config.pass_builder()
pass_builder->DeletePass("fc_fuse_pass") // 去除fc_fuse
```

参数：

- `None`

返回：pass_builder

返回类型：`PassStrategy`

##### PredictorPool

```c++
class PredictorPool;
```

`PredictorPool`对`Predictor`进行了简单的封装，通过传入config和thread的数目来完成初始化，在每个线程中，根据自己的线程id直接从池中取出对应的`Predictor`来完成预测过程。

示例：

```c++
Config config;
// init config
int thread_num = 4;

PredictorPool pool(config, thread_num);

auto predictor0 = pool.Retrive(0);
...
auto predictor3 = pool.Retrive(3);
```

###### Retrive(idx)

根据线程id取出该线程对应的Predictor。

参数：

- `idx(int)` - 线程id

返回：线程对应的Predictor

返回类型：`Predictor*`
