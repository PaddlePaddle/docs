# Introduction to C++ Inference API

To make the deployment of inference model more convenient, a set of high-level APIs are provided in Fluid to hide diverse optimization processes in low level.

Details are as follows:

## <a name="Use AnalysisPredictor to perform high-performance inference"> Use AnalysisPredictor to perform high-performance inference</a>
Paddy fluid uses AnalysisPredictor to perform inference. AnalysisPredictor is a high-performance inference engine. Through the analysis of the calculation graph, the engine completes a series of optimization of the calculation graph (such as the integration of OP, the optimization of memory / graphic memory, the support of MKLDNN, TensorRT and other underlying acceleration libraries), which can greatly improve the inference performance.

In order to show the complete inference process, the following is a complete example of using AnalysisPredictor. The specific concepts and configurations involved will be detailed in the following sections.

#### AnalysisPredictor sample

``` c++
#include "paddle_inference_api.h"

namespace paddle {
void CreateConfig(AnalysisConfig* config, const std::string& model_dirname) {
  // load model from disk
  config->SetModel(model_dirname + "/model",  
                   model_dirname + "/params");  
  // config->SetModel(model_dirname);
  // use SetModelBuffer if load model from memory
  // config->SetModelBuffer(prog_buffer, prog_size, params_buffer, params_size);
  config->EnableUseGpu(100 /*init graphic memory by 100MB*/,  0 /*set GPUID to 0*/);

  /* for cpu
  config->DisableGpu();
  config->EnableMKLDNN();   // enable MKLDNN
  config->SetCpuMathLibraryNumThreads(10);
  */

  config->SwitchUseFeedFetchOps(false);
  // set to true if there are multiple inputs
  config->SwitchSpecifyInputNames(true);
  config->SwitchIrDebug(true);         // If the visual debugging option is enabled, a dot file will be generated after each graph optimization process
  // config->SwitchIrOptim(false);     // The default is true. Turn off all optimizations if set to false
  // config->EnableMemoryOptim();     // Enable memory / graphic memory reuse
}

void RunAnalysis(int batch_size, std::string model_dirname) {
  // 1. create AnalysisConfig
  AnalysisConfig config;
  CreateConfig(&config, model_dirname);

  // 2. create predictor based on config, and prepare input data
  auto predictor = CreatePaddlePredictor(config);
  const channels = 3;
  const height = 224;
  const width = 224;
  std::vector<float> input(batch_size * channels * height * width, 0.f);

  // 3. build inputs
  // uses ZeroCopy API here to avoid extra copying from CPU, improving performance
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(input.data());

  // 4. run inference
  CHECK(predictor->ZeroCopyRun());

  // 5. get outputs
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
  // the model can be downloaded from http://paddle-inference-dist.cdn.bcebos.com/tensorrt_test/mobilenet.tar.gz
  paddle::RunAnalysis(1, "./mobilenet");
  return 0;
}

```

## <a name="Use AnalysisConfig to manage inference configurations"> Use AnalysisConfig to manage inference configurations</a>

AnalysisConfig manages the inference configuration of AnalysisPredictor, providing model path setting, inference engine running device selection, and a variety of options to optimize the inference process. The configuration method is as follows:

#### General optimizing configuration
``` c++
config->SwitchIrOptim(true);  // Enable analysis and optimization of calculation graph,including OP fusion, etc
config->EnableMemoryOptim();  // Enable memory / graphic memory reuse
```
**Note:** Using ZeroCopyTensor requires following setting:
``` c++
config->SwitchUseFeedFetchOps(false);  // disable feed and fetch OP
```

#### set model and param path
When loading the model from disk, there are two ways to set the path of AnalysisConfig to load the model and parameters according to the storage mode of the model and parameter file:

* Non combined form: when there is a model file and multiple parameter files under the model folder 'model_dir', the path of the model folder is passed in. The default name of the model file is'__model_'.
``` c++
config->SetModel("./model_dir");
```

* Combined form: when there is only one model file 'model' and one parameter file 'params' under the model folder' model_dir ', the model file and parameter file path are passed in.
``` c++
config->SetModel("./model_dir/model", "./model_dir/params");
```

At compile time, it is proper to co-build with `libpaddle_fluid.a/.so` .

#### Configure CPU inference

``` c++
config->DisableGpu();          // disable GPU
config->EnableMKLDNN();            // enable MKLDNN, accelerating CPU inference  
config->SetCpuMathLibraryNumThreads(10);        // set number of threads of CPU Math libs, accelerating CPU inference if CPU cores are adequate
```
#### Configure GPU inference
``` c++
config->EnableUseGpu(100, 0); // initialize 100M graphic memory, using GPU ID 0
config->GpuDeviceId();        // Returns the GPU ID being used
// Turn on TRT to improve GPU performance. You need to use library with tensorrt
config->EnableTensorRtEngine(1 << 20             /*workspace_size*/,  
                             batch_size        /*max_batch_size*/,  
                             3                 /*min_subgraph_size*/,
                                AnalysisConfig::Precision::kFloat32 /*precision*/,
                             false             /*use_static*/,
                             false             /*use_calib_mode*/);
```
## <a name="Use ZeroCopyTensor to manage I/O"> Use ZeroCopyTensor to manage I/O</a>

ZeroCopyTensor is the input / output data structure of AnalysisPredictor. The use of zerocopytensor can avoid redundant data copy when preparing input and obtaining output, and improve inference performance.  

**Note:** Using zerocopytensor, be sure to set `config->SwitchUseFeedFetchOps(false);`.

``` c++
// get input/output tensor
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputTensor(input_names[0]);
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputTensor(output_names[0]);

// reshape tensor
input_t->Reshape({batch_size, channels, height, width});

// Through the copy_from_cpu interface, the CPU data is prepared; through the copy_to_cpu interface, the output data is copied to the CPU
input_t->copy_from_cpu<float>(input_data /*data pointer*/);
output_t->copy_to_cpu(out_data /*data pointer*/);

// set LOD
std::vector<std::vector<size_t>> lod_data = {{0}, {0}};
input_t->SetLoD(lod_data);

// get Tensor data pointer
float *input_d = input_t->mutable_data<float>(PaddlePlace::kGPU);  // use PaddlePlace::kCPU when running inference on CPU
int output_size;
float *output_d = output_t->data<float>(PaddlePlace::kGPU, &output_size);
```

## <a name="C++ inference sample"> C++ inference sample</a>
1. Download or compile C++ Inference Library, refer to [Install and Compile C++ Inference Library](./build_and_install_lib_en.html).
2. Download [C++ inference sample](https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/paddle_inference_sample_v1.7.tar.gz) and uncompress it , then enter `sample/inference` directory.  

    `inference` directory structure is as following:

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

    - `mobilenet_test.cc` is the source code for single-thread inference.
    - `thread_mobilenet_test.cc` is the source code for multi-thread inference.
    - `mobilenetv1` is the model directory.
    - `run.sh` is the script for running inference.

3. Configure script:

    Before running, we need to configure script `run.sh` as following:

    ``` shell
    # set whether to enable MKL, GPU or TensorRT. Enabling TensorRT requires WITH_GPU being ON
    WITH_MKL=ON
    WITH_GPU=OFF
    USE_TENSORRT=OFF

    # set path to CUDA lib dir, CUDNN lib dir, TensorRT root dir and model dir
    LIB_DIR=YOUR_LIB_DIR
    CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
    CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
    TENSORRT_ROOT_DIR=YOUR_TENSORRT_ROOT_DIR
    MODEL_DIR=YOUR_MODEL_DIR
    ```

    Please configure `run.sh` depending on your environment.

4. Build and run the sample.  

    ``` shell
    sh run.sh
    ```

## <a name="Performance tuning"> Performance tuning</a>
### Tuning on CPU
1. If the CPU model allows, try to use the version with AVX and MKL.
2. You can try to use Intel's MKLDNN acceleration.
3. When the number of CPU cores available is enough, you can increase the num value in the setting `config->SetCpuMathLibraryNumThreads(num);`.

### Tuning on GPU
1. You can try to open the TensorRT subgraph acceleration engine. Through the graph analysis, Paddle can automatically fuse certain subgraphs, and call NVIDIA's TensorRT for acceleration. For details, please refer to [Use Paddle-TensorRT Library for inference](../../performance_improving/inference_improving/paddle_tensorrt_infer_en.html)。

### Tuning with multi-thread
Paddle Fluid supports optimizing prediction performance by running multiple AnalysisPredictors on different threads, and supports CPU and GPU environments.

sample of using multi-threads is `thread_mobilenet_test.cc` downloaded from [sample](https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/paddle_inference_sample_v1.7.tar.gz). You can change `mobilenet_test` in `run.sh` to `thread_mobilenet_test` to run inference with multi-thread.

```
sh run.sh
```
