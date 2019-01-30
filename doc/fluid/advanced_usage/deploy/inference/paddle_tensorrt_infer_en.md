# Use TensorRT library to inference

NVIDIA TensorRT is a deep learning inference library with high performance, bringing low reply and high QPS for deep learning inference application.
Subgraph is used in Paddle 1.0 to preliminarily integrate TensorRT which we can use to upgrade inference performance of Paddle. The module is still under development. Currently-supported models are AlexNet, MobileNet, ResNet50, VGG19, ResNext, Se-ReNext, GoogleNet, DPN, ICNET, MobileNet-SSD and so on. We will introduce the obtain, usage and pricipe of Paddle-TensorRT in this documentation.


## Build inference libraries with `TensorRT`

**Use Docker to build inference libraries**         

1. Download Paddle  
 
	```
	git clone https://github.com/PaddlePaddle/Paddle.git
	```
	
2. Get docker mirror
  
	```
	nvidia-docker run --name paddle_trt -v $PWD/Paddle:/Paddle -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
	```
 
3. Build Paddle TensorRT       

	```
	# perform following operations in docker container
	cd /Paddle
	mkdir build
	cd build
	# TENSORRT_ROOT is path of TRT, /usr by default,which could be modified according to your requirements
	# MKL can be on according to your requirements
	cmake .. \
	      -DWITH_FLUID_ONLY=ON \
	      -DWITH_MKL=OFF \
	      -DWITH_MKLDNN=OFF \
	      -DCMAKE_BUILD_TYPE=Release \
	      -DWITH_PYTHON=OFF   \
	      -DTENSORRT_ROOT=/usr \
	      -DON_INFER=ON
	
	# build    
	make -j
	# generate inference library
	make inference_lib_dist -j
	```
	
	the directory of library after building is as follows:
	
	```
	fluid_inference_install_dir
	├── paddle
	│      
	├── CMakeCache.txt
	├── version.txt
	├── third_party
	    ├── boost
	    ├── install
	    └── engine3 
	```
   
	Under `fluid_inference_install_dir`,head file and lib of inference library are under paddle directory.
    version.txt contains information of version and configuration of lib while third_party contains third-party libraries inference libraries depend.

## Usage of Paddle TensorRT

[`paddle_inference_api.h`]('https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/paddle_inference_api.h') defines all APIs with TensorRT used. 

In general steps are as follows:
1. Create appropriate AnalysisConfig.    
2. Create `PaddlePredictor` according to config.    
3. Create input tensor.   
4. Get output tensor and show result.   

A complete process is shown below:

```c++
#include "paddle_inference_api.h"

namespace paddle {
using paddle::contrib::AnalysisConfig;

void RunTensorRT(int batch_size, std::string model_dirname) {
  // 1. Create MixedRTConfig
  AnalysisConfig config(true);
  config.model_dir = model_dirname;
  config->use_gpu = true;
  config->device = 0;
  config->fraction_of_gpu_memory = 0.15;
  config->EnableTensorRtEngine(1 << 20 /*work_space_size*/, batch_size /*max_batch_size*/);

  // 2. Create predictor according to config
  auto predictor = CreatePaddlePredictor(config);
  // 3. Create input tensor 
  int height = 224;
  int width = 224;
  float data[batch_size * 3 * height * width] = {0};

  PaddleTensor tensor;
  tensor.shape = std::vector<int>({batch_size, 3, height, width});
  tensor.data = PaddleBuf(static_cast<void *>(data),
                          sizeof(float) * (batch_size * 3 * height * width));
  tensor.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

  // 4. Create output tensor
  std::vector<PaddleTensor> outputs;
  // 5. Inference
  predictor->Run(paddle_tensor_feeds, &outputs, batch_size);

  const size_t num_elements = outputs.front().data.length() / sizeof(float);
  auto *data = static_cast<float *>(outputs.front().data.data());
  for (size_t i = 0; i < num_elements; i++) { 
    std::cout << "output: " << data[i] << std::endl;
  }
}
}  // namespace paddle

int main() { 
  // Download address for model http://paddle-inference-dist.cdn.bcebos.com/tensorrt_test/mobilenet.tar.gz
  paddle::RunTensorRT(1, "./mobilenet");
  return 0;
}
```

## BUild Sample
1. Download Sample   
	```
	wget http://paddle-inference-dist.cdn.bcebos.com/tensorrt_test/paddle_trt_samples.tar.gz
	```
	
	Directory decompressed is as follows:
	
	```
	sample
	├── CMakeLists.txt
	├── mobilenet_test.cc
	├── thread_mobilenet_test.cc
	├── mobilenetv1
	│   ├── model
	│   └── params
	└── run_impl.sh
	```
	
	- `mobilenet_test.cc` is single-thread program file  
	- `thread_mobilenet_test.cc` is multi-thread program file  
	- `mobilenetv1` is model file   

	Supposing the path of inference library is ``BASE_DIR/fluid_inference_install_dir/`` and samples are under the directory ``SAMPLE_BASE_DIR/sample`` 

2. Samples of Building   

	```shell
	cd SAMPLE_BASE_DIR/sample
	# sh run_impl.sh {address of inference libraries} {name of test script} {directory of model}
	sh run_impl.sh BASE_DIR/fluid_inference_install_dir/  mobilenet_test SAMPLE_BASE_DIR/sample/mobilenetv1
	```

3. Build multi-thread samples

 	```shell
	cd SAMPLE_BASE_DIR/sample
	# sh run_impl.sh {address of inference libraries} {name of test script} {directory of model}
	sh run_impl.sh BASE_DIR/fluid_inference_install_dir/  thread_mobilenet_test SAMPLE_BASE_DIR/sample/mobilenetv1
	```


## Subgraph Operation Principle
   Subgraph is used to integrate TensorRT in PaddlePaddle. After the load of model, network can be represented as computing graph composed by variables and computing nodes. Functions Paddle TensorRT implements are to scan the whole picture, find subgraphs that can be optimized with TensorRT and replace them with TensorRT nodes. During the inference of model, Paddle will call TensorRT library to optimize TensorRT nodes call native library of Paddle to optimize other nodes. During the inference, TensorRT can integrate Op horizonally and vertically to filter superfluous Ops and can choose appropriate kenel for specific Op in specific platform to perform optimization so as to speed up inference of model.
   
A simple model expresses the process shown in the picture below: 

**Original Network**
<p align="center">
 <img src="https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_original.png" width="600">
</p>

**Transformed Network**
<p align="center">
 <img src="https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_trt.png" width="600">
</p>

  We can see in the original model network that green nodes represent nodes supported by TensorRT, red nodes represent variables in network and yellow nodes represent nodes can only be operated by native functions in Paddle. Green nodes in original network are extracted to compose subgraph which is replaced by a TensorRT node and transformed as `block-25` node in network. TensorRT library will be called to operate the node during the runtime of network in Paddle.
   






