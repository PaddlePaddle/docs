# C 预测 API介绍

Fluid提供了高度优化的[C++预测库](./native_infer.html)，为了方便使用，我们也提供了封装C++预测库对应的C接口，下面是详细的使用说明。



## C预测相关数据结构

使用C预测API与C++预测API不完全一样，C预测主要包括`PD_Tensor`, `PD_DataType`, `PD_AnalysisConfig`和`PD_PaddleBuf`，分别对应于C++ API中的`PaddleTensor`，`PaddleDType`，`AnalysisConfig`和`PaddleBuf`。另外，C预测API提供了`PD_ZeroCopyData`。接下来将会进一步介绍。本次的Release的C-API目前暂时只支持单输入单输出的模型，后续我们将会继续提供更多的支持。

### PD_Tensor

`PD_Tensor`是预测库输入和输出的数据结构，包括以下函数: 

* `PD_Tensor* PD_NewPaddleTensor()`: 新建一个`PD_Tensor`的指针。
* `void PD_DeletePaddleTensor(PD_Tensor* tensor)`: 删除一个`PD_Tensor`的指针。
* `void PD_SetPaddleTensorName(PD_Tensor* tensor, char* name)`: 指定输入的名称，`name`是一个`char`的指针用于指定名称。
* `void PD_SetPaddleTensorShape(PD_Tensor* tensor, int* shape, int size)`: 设置Tensor的shape。
* `void PD_SetPaddleTensorData(PD_Tensor* tensor, PD_PaddleBuf* buf)`: 设置Tensor的数据。是作为`PD_PaddleBuf`传入的。`PD_PaddleBuf`的接口后文将会介绍。
* `void PD_SetPaddleTensorDType(PD_Tensor* tensor, PD_DataType dtype)`: 设置Tensor的类型。输入是一个枚举类型`PD_DataType`。`PD_DataType`后文将会详细介绍。

同时，提供了读取`PD_Tensor`中以上属性的方法。

* `char* PD_GetPaddleTensorName(const PD_Tensor* tensor)`: 获取`PD_Tensor`的名称`name`。
* `int* PD_GetPaddleTensorShape(const PD_Tensor* tensor, int** size)`: 获取`PD_Tensor`的shape。返回是一个int的指针，另外，通过传参的方式可以得到int指针的大小，即`size`。
* `PD_PaddleBuf* PD_GetPaddleTensorData(const PD_Tensor* tensor)`: 获取`PD_Tensor`的数据。返回值是作为`PD_PaddleBuf`的指针存储的。
* `PD_DataType PD_GetPaddleTensorDType(const PD_Tensor* tensor)`: 获取`PD_Tensor`的类型。返回的是一个枚举类型`PD_DataType`。

### PD_PaddleBuf

`PD_PaddleBuf`存储了上文提到的`PD_Tensor`的数据，包括以下函数: 

* `PD_PaddleBuf* PD_NewPaddleBuf()`: 新建一个`PD_PaddleBuf`的指针。
* `void PD_DeletePaddleTensor(PD_Tensor* tensor)`: 删除一个`PD_PaddleBuf`的指针。
* `void PD_PaddleBufResize(PD_PaddleBuf* buf, size_t length)`: 重新设置`PD_PaddleBuf`指针指向的数据的长度大小。
* `void PD_PaddleBufReset(PD_PaddleBuf* buf, void* data, size_t length)`: 重新设置`PD_PaddleBuf`指针指向的数据的数据本身，也可以用作`PD_PaddleBuf`指针数据的初始化赋值。
* `bool PD_PaddleBufEmpty(PD_PaddleBuf* buf)`: 判断一个`PD_PaddleBuf`的指针指向的数据是否为空。
* `void* PD_PaddleBufData(PD_PaddleBuf* buf)`: 返回一个`PD_PaddleBuf`的指针指向的数据的结果，用void*表示，返回之后，用户可以自行转换成相应的数据类型。
* `size_t PD_PaddleBufLength(PD_PaddleBuf* buf)`: 返回一个`PD_PaddleBuf`的指针指向的数据的长度大小。

### PD_DataType

`PD_DataType`是一个提供给用户的枚举，用于设定存有用户数据的`PD_Tensor`的数据类型。包括以下成员：

* `PD_FLOAT32`: 32位浮点型
* `PD_INT32`: 32位整型
* `PD_INT64`: 64位整型
* `PD_UINT8`: 8位无符号整型

#### 代码示例
首先可以新建一个`PD_Tensor`和一个`PD_PaddleBuf`的指针。
``` C
PD_Tensor* input = PD_NewPaddleTensor();
PD_PaddleBuf* buf = PD_NewPaddleBuf();
```
调用设置`PD_PaddleBuf`的函数调用如下: 
``` C
int batch = 1;
int channel = 3;
int height = 318;
int width = 318;
int shape[4] = {batch, channel, height, width};
int shape_size = 4;
float* data = (float *) malloc(sizeof(float) * (batch * channel * height * width));
if (PD_PaddleBufEmpty(buf))
  PD_PaddleBufReset(buf, (void *)(data),
                  sizeof(float) * (batch * channel * height * width));
float* data__ = (float *) PD_PaddleBufData(buf);
size_t length__ = PD_PaddleBufLength(buf);
```
设置了`PD_PaddleBuf`之后，就可以顺利完成对`PD_Tensor`的设置。
``` C
char* name = "image";
PD_SetPaddleTensorName(input, name);
PD_SetPaddleTensorDType(input, PD_DataType::PD_FLOAT32);
PD_SetPaddleTensorShape(input, shape, shape_size);
PD_SetPaddleTensorData(input, buf);
```

### PD_AnalysisConfig

`PD_AnalysisConfig`是创建预测引擎的配置，提供了模型路径设置、预测引擎运行设备选择以及多种优化预测流程的选项，主要包括以下方法  

* `PD_AnalysisConfig* PD_NewAnalysisConfig()`: 新建一个`PD_AnalysisConfig`的指针。
* `void PD_DeleteAnalysisConfig(PD_AnalysisConfig* config)`: 删除一个`PD_AnalysisConfig`的指针。
* `void PD_SetModel(PD_AnalysisConfig* config, const char* model_dir, const char* params_path = NULL)`: 设置模型的路径，输入的参数包括`PD_AnalysisConfig`，`model_dir`，`params_path`，其中`model_dir`是指的是模型保存位置的路径，一般不用包括文件名，`params_path`为可选参数，<strong>注意</strong>: 
    - 如果不给定`params_path`，则认为该模型的参数存储路径与`model_dir`一致，且模型文件和参数文件是按照默认的文件名存储的，此时参数文件可能有多个。
    - 如果提供了`params_path`，为了方便用户的自定义，则在指明`model_dir`路径最后需要加上模型文件的文件名传入。
* `const char* PD_ModelDir(const PD_AnalysisConfig* config)`: 如果未指明`PD_SetModel()`的`params_path`，则可以返回模型文件夹路径。
* `const char* PD_ProgFile(const PD_AnalysisConfig* config)`: 如果是指明`PD_SetModel()`的`params_path`，则可以返回模型文件路径。
* `const char* PD_ParamsFile(const PD_AnalysisConfig* config)`: 如果是指明`PD_SetModel()`的`params_path`，则可以返回参数文件路径。
* `void PD_EnableUseGpu(PD_AnalysisConfig* config, uint64_t memory_pool_init_size_mb, int device_id = 0)`: 设置开启GPU，并且设定GPU显存(单位M)和设备的Device ID。
* `void PD_DisableGpu(PD_AnalysisConfig* config)`: 禁用GPU。
* `int PD_GpuDeviceId(const PD_AnalysisConfig* config)`: 返回使用的GPU设备的ID。
* `void PD_SwitchIrOptim(PD_AnalysisConfig* config, bool x = true)`: IR优化(默认开启)。
* `void PD_EnableTensorRtEngine(PD_AnalysisConfig* config, int workspace_size = 1 << 20, int max_batch_size = 1, int min_subgraph_size = 3, Precision precision = Precision::kFloat32, bool use_static = false, bool use_calib_mode = false)`: 开启TensorRT。关于参数的解释，详见``使用Paddle-TensorRT库预测``。
* `void PD_EnableMKLDNN(PD_AnalysisConfig* config)`: 开启MKLDNN。

#### 代码示例
首先，新建一个`PD_AnalysisConfig`的指针。
``` C
PD_AnalysisConfig* config = PD_NewAnalysisConfig();
```
如前文所述，设置模型和参数路径有两种形式：
* 当模型文件夹下存在一个以默认文件名保存的模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为`__model__`
``` C
const char* model_dir = "./model";
PD_SetModel(config, model_dir);
```
* 当模型文件夹下只有一个模型文件和一个参数文件，传入模型文件和参数文件
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
```

### PD_PredictorRun

``` C
bool PD_PredictorRun(const PD_AnalysisConfig* config, PD_Tensor* inputs, int in_size, PD_Tensor** output_data, int* out_size, int batch_size);
```

`PD_PredictorRun`是运行预测的引擎。输入的参数解释如下：
`config`(`PD_AnalysisConfig`): 用于设置预测的配置变量。
`inputs`(`PD_Tensor*`): 输入的`PD_Tensor`的指针。如果有多个`PD_Tensor`作为输入，则`inputs`将指向第一个，其数组的大小由`in_size`决定。
`in_size`(`int`): 输入的`PD_Tensor`的个数。
`output_data`(`PD_Tensor*`): 输出的`PD_Tensor`的指针。
`out_size`(`int**`): 输出的`PD_Tensor`的个数。是指针的指针，需要新建后传值进入函数，函数执行完会得到赋值，即，获得输出的`PD_Tensor`的个数。
`batch_size`(`int`): `batch_size`的大小。

#### 代码示例

如前文所述，当完成网络配置`PD_AnalysisConfig`以及`input`设置之后，只需要简单的几行代码就可以获得模型的输出。假设模型输入的个数是`in_size = 1`。

``` C
int in_size = 1;
int batch_size = 1;

PD_Tensor* output = PD_NewPaddleTensor();
int output_size;

PD_PredictorRun(config, input, in_size, &output, &output_size, batch_size);
```

## 完整使用示例

下面是使用Fluid C API进行预测的一个完整示例，使用resnet50模型

下载[resnet50模型](http://paddle-inference-dist.bj.bcebos.com/resnet50_model.tar.gz)并解压，运行如下命令将会调用预测引擎。

``` C
#include "c_api.h"

int main() {
    PD_AnalysisConfig* config = PD_NewAnalysisConfig();
    PD_DisableGpu(config);
    const char* model_path = "./model/model";
    const char* params_path = "./params/params";
    PD_SetModel(config, model_path, params_path);
    
    int in_size = 1;
    PD_Tensor* input = PD_NewPaddleTensor();
    PD_PaddleBuf* buf = PD_NewPaddleBuf();
    int batch = 1;
    int channel = 3;
    int height = 318;
    int width = 318;
    int shape[4] = {batch, channel, height, width};
    int shape_size = 4;
    float* data = (float *) malloc(sizeof(float) * (batch * channel * height * width));
    PD_PaddleBufReset(buf, (void *)(data),
                    sizeof(float) * (batch * channel * height * width));

    char* name[5] = {'d', 'a', 't', 'a', '\0'};
    PD_SetPaddleTensorName(input, name);
    PD_SetPaddleTensorDType(input, PD_DataType::PD_FLOAT32);
    PD_SetPaddleTensorShape(input, shape, shape_size);
    PD_SetPaddleTensorData(input, buf);

    PD_Tensor* output = PD_NewPaddleTensor();
    
    // 注意output_size和out_size，前者是表示输出的PD_Tensor有多少个output输出，后者是指的当前的一个输出的out_data的size大小
    int output_size;
    PD_PredictorRun(config, input, in_size, &output, &output_size, batch);

    const char* output_name = PD_GetPaddleTensorName(output);
    PD_PaddleBuf* out_buf = PD_GetPaddleTensorData(output);
    float* out_data = (float *) PD_PaddleBufData(out_buf);
    size_t out_size = PD_PaddleBufLength(out_buf) / sizeof(float);
    return 0;
}
```
