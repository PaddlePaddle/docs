# C++ 预测 API介绍

为了更简单方便地预测部署，PaddlePaddle 提供了一套高层 C++ API 预测接口。

下面是详细介绍。


## 内容

- [使用AnalysisPredictor进行高性能预测](#使用AnalysisPredictor进行高性能预测)
- [使用AnalysisConfig管理预测配置](#使用AnalysisConfig管理预测配置)
- [使用ZeroCopyTensor管理输入/输出](#使用ZeroCopyTensor管理输入/输出)
- [C++预测样例编译测试](#C++预测样例编译测试)
- [性能调优](#性能调优)



## <a name="使用AnalysisPredictor进行高性能预测"> 使用AnalysisPredictor进行高性能预测</a>
Paddle Fluid采用 AnalysisPredictor 进行预测。AnalysisPredictor 是一个高性能预测引擎，该引擎通过对计算图的分析，完成对计算图的一系列的优化（如OP的融合、内存/显存的优化、 MKLDNN，TensorRT 等底层加速库的支持等），能够大大提升预测性能。

为了展示完整的预测流程，下面是一个使用 AnalysisPredictor 进行预测的完整示例，其中涉及到的具体概念和配置会在后续部分展开详细介绍。

#### AnalysisPredictor 预测示例

``` c++
#include "paddle_inference_api.h"

namespace paddle {
void CreateConfig(AnalysisConfig* config, const std::string& model_dirname) {
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

  // 使用ZeroCopyTensor，此处必须设置为false
  config->SwitchUseFeedFetchOps(false);
  // 若输入为多个，此处必须设置为true
  config->SwitchSpecifyInputNames(true);
  config->SwitchIrDebug(true); 		// 可视化调试选项，若开启，则会在每个图优化过程后生成dot文件
  // config->SwitchIrOptim(false); 	// 默认为true。如果设置为false，关闭所有优化
  // config->EnableMemoryOptim(); 	// 开启内存/显存复用
}

void RunAnalysis(int batch_size, std::string model_dirname) {
  // 1. 创建AnalysisConfig
  AnalysisConfig config;
  CreateConfig(&config, model_dirname);

  // 2. 根据config 创建predictor，并准备输入数据，此处以全0数据为例
  auto predictor = CreatePaddlePredictor(config);
  const int channels = 3;
  const int height = 224;
  const int width = 224;
  std::vector<float> input(batch_size * channels * height * width, 0.f);

  // 3. 创建输入
  // 使用了ZeroCopy接口，可以避免预测中多余的CPU copy，提升预测性能
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu(input.data());

  // 4. 运行预测引擎
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

## <a name="使用AnalysisConfig管理预测配置"> 使用AnalysisConfig管理预测配置</a>

AnalysisConfig管理AnalysisPredictor的预测配置，提供了模型路径设置、预测引擎运行设备选择以及多种优化预测流程的选项。配置方法如下：

#### 通用优化配置
``` c++
config->SwitchIrOptim(true);  // 开启计算图分析优化，包括OP融合等
config->EnableMemoryOptim();  // 开启内存/显存复用
```
**Note:** 使用ZeroCopyTensor必须设置：
``` c++
config->SwitchUseFeedFetchOps(false);  // 关闭feed和fetch OP使用，使用ZeroCopy接口必须设置此项
```

#### 设置模型和参数路径
从磁盘加载模型时，根据模型和参数文件存储方式不同，设置AnalysisConfig加载模型和参数的路径有两种形式：

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
config->DisableGpu();		  // 禁用GPU
config->EnableMKLDNN();	  	  // 开启MKLDNN，可加速CPU预测
config->SetCpuMathLibraryNumThreads(10); 	   // 设置CPU Math库线程数，CPU核心数支持情况下可加速预测
```
#### 配置GPU预测
``` c++
config->EnableUseGpu(100, 0); // 初始化100M显存，使用GPU ID为0
config->GpuDeviceId();        // 返回正在使用的GPU ID
// 开启TensorRT预测，可提升GPU预测性能，需要使用带TensorRT的预测库
config->EnableTensorRtEngine(1 << 20      	   /*workspace_size*/,  
                        	 batch_size        /*max_batch_size*/,  
                        	 3                 /*min_subgraph_size*/,
                       		 AnalysisConfig::Precision::kFloat32 /*precision*/,
                        	 false             /*use_static*/,
                        	 false             /*use_calib_mode*/);
```


## <a name="使用ZeroCopyTensor管理输入/输出"> 使用ZeroCopyTensor管理输入/输出</a>

ZeroCopyTensor是AnalysisPredictor的输入/输出数据结构。ZeroCopyTensor的使用可以避免预测时候准备输入以及获取输出时多余的数据copy，提高预测性能。  

**Note:** 使用ZeroCopyTensor，务必在创建config时设置`config->SwitchUseFeedFetchOps(false);`。

``` c++
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

// 获取Tensor数据指针
float *input_d = input_t->mutable_data<float>(PaddlePlace::kGPU);  // CPU下使用PaddlePlace::kCPU
int output_size;
float *output_d = output_t->data<float>(PaddlePlace::kGPU, &output_size);
```

## <a name="C++预测样例编译测试"> C++预测样例编译测试</a>
1. 下载或编译paddle预测库，参考[安装与编译C++预测库](./build_and_install_lib_cn.html)。
2. 下载[预测样例](https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/paddle_inference_sample_v1.7.tar.gz)并解压，进入`sample/inference`目录下。   

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
Paddle Fluid支持通过在不同线程运行多个AnalysisPredictor的方式来优化预测性能，支持CPU和GPU环境。

使用多线程预测的样例详见[C++预测样例编译测试](#C++预测样例编译测试)中下载的[预测样例](https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/paddle_inference_sample_v1.7.tar.gz)中的
`thread_mobilenet_test.cc`文件。可以将`run.sh`中`mobilenet_test`替换成`thread_mobilenet_test`再执行

```
sh run.sh
```

即可运行多线程预测样例。
