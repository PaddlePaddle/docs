# Use TensorRT Library for inference

NVIDIA TensorRT is a is a platform for high-performance deep learning inference. It delivers low latency and high throughput for deep learning inference application.
Subgraph is used in Paddle 1.0 to preliminarily integrate TensorRT, which enables TensorRT module to enhance inference performance of paddle models. The module is still under development. Currently supported models are AlexNet, MobileNet, ResNet50, VGG19, ResNext, Se-ReNext, GoogleNet, DPN, ICNET, MobileNet-SSD and so on. We will introduce the obtaining, usage and theory of Paddle-TensorRT library in this documentation.


## Build inference libraries with `TensorRT`

**Use Docker to build inference libraries**         

1. Download Paddle  
 
	```
	git clone https://github.com/PaddlePaddle/Paddle.git
	```
	
2. Get docker image
  
	```
	nvidia-docker run --name paddle_trt -v $PWD/Paddle:/Paddle -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
	```
 
3. Build Paddle TensorRT       

	```
	# perform the following operations in docker container
	cd /Paddle
	mkdir build
	cd build
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

## Usage of Paddle TensorRT

[`paddle_inference_api.h`]('https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/paddle_inference_api.h') defines all APIs of TensorRT. 

General steps are as follows:
1. Create appropriate AnalysisConfig.    
2. Create `PaddlePredictor` based on config.    
3. Create input tensor.   
4. Get output tensor and output result.   

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

  // 2. Create predictor based on config
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
  // Download address of the model http://paddle-inference-dist.cdn.bcebos.com/tensorrt_test/mobilenet.tar.gz
  paddle::RunTensorRT(1, "./mobilenet");
  return 0;
}
```
The compilation process is [here](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/inference/api/demo_ci)


## Subgraph Theory

   Subgraph is used to integrate TensorRT in PaddlePaddle. After model is loaded, neural network can be represented as a computing graph composed of variables and computing nodes. Functions Paddle TensorRT implements are to scan the whole picture, discover subgraphs that can be optimized with TensorRT and replace them with TensorRT nodes. During the inference of model, Paddle will call TensorRT library to optimize TensorRT nodes and call native library of Paddle to optimize other nodes. During the inference, TensorRT can integrate Op horizonally and vertically to filter redundant Ops and is able to choose appropriate kernel for specific Op in specific platform to speed up the inference of model.
   
A simple model expresses the process : 

**Original Network**
<p align="center">
 <img src="https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_original.png" width="600">
</p>

**Transformed Network**
<p align="center">
 <img src="https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_trt.png" width="600">
</p>

  We can see in the Original Network that the green nodes represent nodes supported by TensorRT, the red nodes represent variables in network and yellow nodes represent nodes which can only be operated by native functions in Paddle. Green nodes in original network are extracted to compose subgraph which is replaced by a single TensorRT node to be transformed into `block-25` node in network. When such nodes are encountered during the runtime, TensorRT library will be called to execute them.
   

