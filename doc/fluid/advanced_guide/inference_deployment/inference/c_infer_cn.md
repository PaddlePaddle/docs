# C 预测 API介绍

Fluid提供了高度优化的[C++预测库](./native_infer.html)，为了方便使用，我们也提供了封装了C++预测库对应的C接口。C接口的使用方式，首先是需要`#include paddle_c_api.h`，头文件`paddle_c_api.h`可以在Paddle的仓库中的`paddle/fluid/inference/capi/paddle_c_api.h`找到，或是在编译Paddle的`Paddle/build/`路径下，`build/fluid_inference_c_install_dir/paddle/include/`路径下找到。此外，使用 CAPI 还需要在编译项目的时候，链接相关的编译的库`libpaddle_fluid_c.so`。下面是详细的使用说明。

需要说明的是，与 C++ API 不同，C API 为了兼顾多语言封装的需要，将不会再设置默认参数，即使用时，所有的参数都需要用户显式地提供。


## C预测相关数据结构

使用C预测API与C++预测API不完全一样，C预测主要包括`PD_AnalysisConfig`, `PD_DataType`, `PD_Predictor`, `PD_Buffer`和`PD_ZeroCopyTensor`。接下来将会进一步详细地介绍这些数据结构以及使用的方法，并提供相应的示例。

### PD_AnalysisConfig

`PD_AnalysisConfig`是创建预测引擎的配置，提供了模型路径设置、预测引擎运行设备选择以及多种优化预测流程的选项，主要包括以下方法  

* `PD_AnalysisConfig* PD_NewAnalysisConfig()`: 新建一个`PD_AnalysisConfig`的指针。
* `void PD_DeleteAnalysisConfig(PD_AnalysisConfig* config)`: 删除一个`PD_AnalysisConfig`的指针。
* `void PD_SetModel(PD_AnalysisConfig* config, const char* model_dir, const char* params_path)`: 设置模型的路径，输入的参数包括`PD_AnalysisConfig`，`model_dir`，`params_path`，其中`model_dir`是指的是模型保存位置的路径，一般不用包括文件名，`params_path`为可选参数，<strong>注意</strong>: 
    - 如果不给定`params_path`，即`params_path`为`NULL`，则认为该模型的参数存储路径与`model_dir`一致，且模型文件和参数文件是按照默认的文件名存储的，此时参数文件可能有多个。此时，需要用户输入参数与模型文件的`model_dir`，即<strong>模型和参数保存的路径名</strong>，不需要指定文件名，同时，需要显式地设置`params_path`为`NULL`。
    - 如果提供了`params_path`，为了方便用户的自定义，则在指明`model_dir`路径最后需要加上模型文件的文件名传入，即`model_dir`传入对应的<strong>模型文件的路径</strong>，`params_path`传入对应的<strong>模型参数文件的路径</strong>，需要指定文件名。
* `const char* PD_ModelDir(const PD_AnalysisConfig* config)`: 如果未指明`PD_SetModel()`的`params_path`，则可以返回模型文件夹路径。
* `const char* PD_ProgFile(const PD_AnalysisConfig* config)`: 如果是指明`PD_SetModel()`的`params_path`，则可以返回模型文件路径。
* `const char* PD_ParamsFile(const PD_AnalysisConfig* config)`: 如果是指明`PD_SetModel()`的`params_path`，则可以返回参数文件路径。
* `void PD_SwitchSpecifyInputNames(PD_AnalysisConfig* config, bool x)`: 设置为`true`是指模型运算在读取输入的时候，依据名称来确定不同的输入，否则根据输入的顺序。使用`PD_ZeroCopyTensor`并且是多输入的情况，建议设置为`true`。
* `void PD_SwitchUseFeedFetchOps(PD_AnalysisConfig* config, bool x)`: 设置是否使用`feed`，`fetch` op。在使用`PD_ZeroCopyTensor`必须设置该选项为`false`。
* `void PD_EnableUseGpu(PD_AnalysisConfig* config, uint64_t memory_pool_init_size_mb, int device_id)`: 设置开启GPU，并且设定GPU显存(单位M)和设备的Device ID。
* `void PD_DisableGpu(PD_AnalysisConfig* config)`: 禁用GPU。
* `int PD_GpuDeviceId(const PD_AnalysisConfig* config)`: 返回使用的GPU设备的ID。
* `void PD_SwitchIrOptim(PD_AnalysisConfig* config, bool x)`: 设置预测是否开启IR优化。
* `void PD_EnableTensorRtEngine(PD_AnalysisConfig* config, int workspace_size, int max_batch_size, int min_subgraph_size, Precision precision, bool use_static, bool use_calib_mode)`: 开启TensorRT。关于参数的解释，详见[使用Paddle-TensorRT库预测](../../performance_improving/inference_improving/paddle_tensorrt_infer.html)。
* `void PD_EnableMKLDNN(PD_AnalysisConfig* config)`: 开启MKLDNN。

#### 代码示例
首先，新建一个`PD_AnalysisConfig`的指针。
``` C
PD_AnalysisConfig* config = PD_NewAnalysisConfig();
```
如前文所述，设置模型和参数路径有两种形式：
* 当模型文件夹下存在一个以默认文件名保存的模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为`__model__`，需要显式地设置`params_path`为`NULL`，不需要指定文件名。
``` C
const char* model_dir = "./model/";
PD_SetModel(config, model_dir, NULL);
```
* 当模型文件夹下只有一个模型文件和一个参数文件，传入模型文件和参数文件，需要指定文件名。
``` C
const char* model_path = "./model/model";
const char* params_path = "./params/params";
PD_SetModel(config, model_path, params_path);
```

其他预测引擎配置选项示例如下
``` C
PD_EnableUseGpu(config, 100, 0); // 初始化100M显存，使用的gpu id为0
PD_GpuDeviceId(config);          // 返回正在使用的gpu id
PD_DisableGpu(config);           // 禁用gpu
PD_SwitchIrOptim(config, true);  // 开启IR优化
PD_EnableMKLDNN(config);         // 开启MKLDNN
PD_SwitchSpecifyInputNames(config, true);
PD_SwitchUseFeedFetchOps(config, false);
```

### PD_ZeroCopyTensor

`PD_ZeroCopyTensor`是设置数据传入预测运算的数据结构。包括一下成员：

* `data - (PD_Buffer)`: 设置传入数据的值。
* `shape - (PD_Buffer)`: 设置传入数据的形状（shape）。
* `lod - (PD_Buffer)`: 设置数据的`lod`，目前只支持一阶的`lod`。
* `dtype - (PD_DataType)`: 设置传入数据的数据类型，用枚举`PD_DataType`表示。
* `name - (char*)`: 设置传入数据的名称。

涉及使用`PD_ZeroCopyTensor`有以下方法：

* `PD_ZeroCopyTensor* PD_NewZeroCopyTensor()`: 新创建一个`PD_ZeroCopyTensor`的指针。
* `void PD_DeleteZeroCopyTensor(PD_ZeroCopyTensor*)`: 删除一个`PD_ZeroCopyTensor`的指针。
* `void PD_InitZeroCopyTensor(PD_ZeroCopyTensor*)`: 使用默认初始化一个`PD_ZeroCopyTensor`的指针并分配的内存空间。
* `void PD_DestroyZeroCopyTensor(PD_ZeroCopyTensor*)`: 删除`PD_ZeroCopyTensor`指针中，`data`，`shape`，`lod`的`PD_Buffer`的变量。

### PD_DataType

`PD_DataType`是一个提供给用户的枚举，用于设定存有用户数据的`PD_ZeroCopyTensor`的数据类型。包括以下成员：

* `PD_FLOAT32`: 32位浮点型
* `PD_INT32`: 32位整型
* `PD_INT64`: 64位整型
* `PD_UINT8`: 8位无符号整型

#### 代码示例
首先可以新建一个`PD_ZeroCopyTensor`。
``` C
PD_ZeroCopyTensor input;
PD_InitZeroCopyTensor(&input);
```
调用设置`PD_ZeroCopyTensor`的数据类型的方式如下: 
``` C
input.dtype = PD_FLOAT32;
```

### PD_Buffer

`PD_Buffer`可以用于设置`PD_ZeroCopyTensor`数据结构中，数据的`data`，`shape`和`lod`。包括以下成员：

* `data`: 输入的数据，类型是`void*`，用于存储数据开始的地址。
* `length`: 输入数据的实际的<strong>字节长度</strong>。
* `capacity`: 为数据分配的内存大小，必定大于等于`length`。

### 示例代码
``` C
PD_ZeroCopyTensor input;
PD_InitZeroCopyTensor(&input);
// 设置输入的名称
input.name = "data";
// 设置输入的数据大小
input.data.capacity = sizeof(float) * 1 * 3 * 300 * 300;
input.data.length = input.data.capacity;
input.data.data = malloc(input.data.capacity);
// 设置数据的输入的形状 shape
int shape[] = {1, 3, 300, 300};
input.shape.data = (int *)shape;
input.shape.capacity = sizeof(shape);
input.shape.length = sizeof(shape);
// 设置输入数据的类型
input.dtype = PD_FLOAT32;
```

### PD_Predictor

`PD_Predictor`是一个高性能预测引擎，该引擎通过对计算图的分析，可以完成对计算图的一系列的优化（如OP的融合、内存/显存的优化、 MKLDNN，TensorRT 等底层加速库的支持等）。主要包括一下函数：

* `PD_Predictor* PD_NewPredictor(const PD_AnalysisConfig* config)`: 创建一个新的`PD_Predictor`的指针。
* `void PD_DeletePredictor(PD_Predictor* predictor)`: 删除一个`PD_Predictor`的指针。
* `int PD_GetInputNum(const PD_Predictor* predictor)`: 获取模型输入的个数。
* `int PD_GetOutputNum(const PD_Predictor* predictor)`: 获取模型输出的个数。
* `const char* PD_GetInputName(const PD_Predictor* predictor, int n)`: 获取模型第`n`个输入的名称。
* `const char* PD_GetOutputName(const PD_Predictor* predictor, int n)`: 获取模型第`n`个输出的名称。
* `void PD_SetZeroCopyInput(PD_Predictor* predictor, const PD_ZeroCopyTensor* tensor)`: 使用`PD_ZeroCopyTensor`数据结构设置模型输入的具体值、形状、lod等信息。目前只支持一阶lod。
* `void PD_GetZeroCopyOutput(PD_Predictor* predictor, PD_ZeroCopyTensor* tensor)`: 使用`PD_ZeroCopyTensor`数据结构获取模型输出的具体值、形状、lod等信息。目前只支持一阶lod。
* `void PD_ZeroCopyRun(PD_Predictor* predictor)`: 运行预测的引擎，完成模型由输入到输出的计算。

#### 代码示例

如前文所述，当完成网络配置`PD_AnalysisConfig`以及输入`PD_ZeroCopyTensor`的设置之后，只需要简单的几行代码就可以获得模型的输出。

首先完成`PD_AnalysisConfig`的设置，设置的方式与相关的函数如前文所述，这里同样给出了示例。

``` C
PD_AnalysisConfig* config = PD_NewAnalysisConfig();
const char* model_dir = "./model/";
PD_SetModel(config, model_dir, NULL);
PD_DisableGpu(config);
PD_SwitchSpecifyInputNames(config, true); // 使用PD_ZeroCopyTensor并且是多输入建议设置。
PD_SwitchUseFeedFetchOps(config, false);  // 使用PD_ZeroCopyTensor一定需要设置为false。
```

其次，完成相应的输入的设置，设置的方式如前文所述，这里同样给出了示例。

``` C
PD_ZeroCopyTensor input;
PD_InitZeroCopyTensor(&input);
// 设置输入的名称
input.name = (char *)(PD_GetInputName(predictor, 0));
// 设置输入的数据大小
input.data.capacity = sizeof(float) * 1 * 3 * 300 * 300;
input.data.length = input.data.capacity;
input.data.data = malloc(input.data.capacity);
// 设置数据的输入的形状(shape)
int shape[] = {1, 3, 300, 300};
input.shape.data = (int *)shape;
input.shape.capacity = sizeof(shape);
input.shape.length = sizeof(shape);
// 设置输入数据的类型
input.dtype = PD_FLOAT32;
```

最后，执行预测引擎，完成计算的步骤。

``` C
PD_Predictor *predictor = PD_NewPredictor(config);

int input_num = PD_GetInputNum(predictor);
printf("Input num: %d\n", input_num);
int output_num = PD_GetOutputNum(predictor);
printf("Output num: %d\n", output_num);

PD_SetZeroCopyInput(predictor, &input); // 这里只有一个输入，根据多输入情况，可以传入一个数组

PD_ZeroCopyRun(predictor); // 执行预测引擎

PD_ZeroCopyTensor output;
PD_InitZeroCopyTensor(&output);
output.name = (char *)(PD_GetOutputName(predictor, 0));
PD_GetZeroCopyOutput(predictor, &output);
```

最后，可以根据前文所述的`PD_ZeroCopyTensor`的数据结构，获得返回的数据的值等信息。

## 完整使用示例

下面是使用Fluid C API进行预测的一个完整示例，使用resnet50模型

下载[resnet50模型](http://paddle-inference-dist.bj.bcebos.com/resnet50_model.tar.gz)并解压，运行如下代码将会调用预测引擎。

``` C
#include "paddle_c_api.h"
#include <memory.h>
#include <malloc.h>

/*
 * The main procedures to run a predictor according to c-api:
 * 1. Create config to set how to process the inference.
 * 2. Prepare the input PD_ZeroCopyTensor for the inference.
 * 3. Set PD_Predictor. 
 * 4. Call PD_ZeroCopyRun() to start. 
 * 5. Obtain the output. 
 * 6. According to the size of the PD_PaddleBuf's data's size, print all the output data. 
 */
int main() {
    // 配置 PD_AnalysisConfig
    PD_AnalysisConfig* config = PD_NewAnalysisConfig();
    PD_DisableGpu(config);
    const char* model_path = "./model/model";
    const char* params_path = "./model/params";
    PD_SetModel(config, model_path, params_path);
    PD_SwitchSpecifyInputNames(config, true);
    PD_SwitchUseFeedFetchOps(config, false);

    // 新建一个 PD_Predictor 的指针
    PD_Predictor *predictor = PD_NewPredictor(config);
    // 获取输入输出的个数
    int input_num = PD_GetInputNum(predictor);
    printf("Input num: %d\n", input_num);
    int output_num = PD_GetOutputNum(predictor);
    printf("Output num: %d\n", output_num);
    
    // 设置输入的数据结构
    PD_ZeroCopyTensor input;
    PD_InitZeroCopyTensor(&input);
    // 设置输入的名称
    input.name = (char *)(PD_GetInputName(predictor, 0));
    // 设置输入的数据大小
    input.data.capacity = sizeof(float) * 1 * 3 * 318 * 318;
    input.data.length = input.data.capacity;
    input.data.data = malloc(input.data.capacity);
    memset(input.data.data, 0, (sizeof(float) * 3 * 318 * 318));

    // 设置数据的输入的形状(shape)
    int shape[] = {1, 3, 318, 318};
    input.shape.data = (int *)shape;
    input.shape.capacity = sizeof(shape);
    input.shape.length = sizeof(shape);
    // 设置输入数据的类型
    input.dtype = PD_FLOAT32;

    PD_SetZeroCopyInput(predictor, &input);

    // 执行预测引擎
    PD_ZeroCopyRun(predictor);

    // 获取预测输出
    PD_ZeroCopyTensor output;
    PD_InitZeroCopyTensor(&output);
    output.name = (char *)(PD_GetOutputName(predictor, 0));
    // 获取 output 之后，可以通过该数据结构，读取到 data, shape 等信息
    PD_GetZeroCopyOutput(predictor, &output);  

    float* result = (float *)(output.data.data);
    int result_length = output.data.length / sizeof(float);

    return 0;
}
```

运行以上代码，需要将 paddle_c_api.h 拷贝到指定位置，确保编译时可以找到这个头文件。同时，需要将 libpaddle_fluid_c.so 的路径加入环境变量。

最后可以使用 gcc 命令编译。

``` shell
gcc ${SOURCE_NAME} \
    -lpaddle_fluid_c
```
