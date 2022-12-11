# Use Paddle-TensorRT Library for inference

NVIDIA TensorRT is a is a platform for high-performance deep learning inference. It delivers low latency and high throughput for deep learning inference application.
Subgraph is used in PaddlePaddle to preliminarily integrate TensorRT, which enables TensorRT module to enhance inference performance of paddle models. The module is still under development. Currently supported models are as following:

|classification|detection|segmentation|
|---|---|---|
|mobilenetv1|yolov3|ICNET|
|resnet50|SSD||
|vgg16|mask-rcnn||
|resnext|faster-rcnn||
|AlexNet|cascade-rcnn||
|Se-ResNext|retinanet||
|GoogLeNet|mobilenet-SSD||
|DPN|||

We will introduce the obtaining, usage and theory of Paddle-TensorRT library in this documentation.

**Note:**

1. When compiling from source, TensorRT library currently only supports GPU compilation, and you need to set the compilation option TensorRT_ROOT to the path where tensorrt is located.
2. Windows support requires TensorRT version 5.0 or higher.
3. Paddle-TRT currently only supports fixed input shape.
4. After downloading and installing tensorrt, you need to manually add virtual destructors for `class IPluginFactory` and `class IGpuAllocator` in the `NvInfer.h` file:
    ``` c++
    virtual ~IPluginFactory() {};
    virtual ~IGpuAllocator() {};
    ```

## <a name="Paddle-TRT interface usage">Paddle-TRT interface usage</a>

When using AnalysisPredictor, we enable Paddle-TRT by setting

``` c++
config->EnableTensorRtEngine(1 << 20      /* workspace_size*/,
                        batch_size        /* max_batch_size*/,
                        3                 /* min_subgraph_size*/,
                        AnalysisConfig::Precision::kFloat32 /* precision*/,
                        false             /* use_static*/,
                        false             /* use_calib_mode*/);
```
The details of this interface is as following:

- **`workspace_size`**: type:int, default is 1 << 20. Sets the max workspace size of TRT. TensorRT will choose kernels under this constraint.
- **`max_batch_size`**: type:int, default is 1. Sets the max batch size. Batch sizes during runtime cannot exceed this value.
- **`min_subgraph_size`**: type:int, default is 3. Subgraph is used to integrate TensorRT in PaddlePaddle. To avoid low performance, Paddle-TRT is only enabled when th number of nodes in th subgraph is more than `min_subgraph_size`.
- **`precision`**: type:`enum class Precision {kFloat32 = 0, kHalf, kInt8,};`, default is `AnalysisConfig::Precision::kFloat32`. Sets the precision of TRT, supporting FP32(kFloat32), FP16(kHalf), Int8(kInt8). Using Paddle-TRT int8 calibration requires setting `precision` to  `AnalysisConfig::Precision::kInt8`, and `use_calib_mode` to true.
- **`use_static`**: type:bool, default is false. If set to true, Paddle-TRT will serialize optimization information to disk, to deserialize next time without optimizing again.
- **`use_calib_mode`**: type:bool, default is false. Using Paddle-TRT int8 calibration requires setting this option to true.

**Note：** Paddle-TRT currently only supports fixed input shape.

## <a name="Paddle-TRT example compiling test">Paddle-TRT example compiling test</a>

1. Download or compile Paddle Inference with TensorRT support, refer to [Install and Compile C++ Inference Library](../../inference_deployment/inference/build_and_install_lib_en.html).
2. Download NVIDIA TensorRT(with consistent version of cuda and cudnn in local environment) from [NVIDIA TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) with an NVIDIA developer account.
3. Download [Paddle Inference sample](https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/paddle_inference_sample_v1.7.tar.gz) and uncompress, and enter `sample/paddle-TRT` directory.

    `paddle-TRT` directory structure is as following:

    ```
    paddle-TRT
    ├── CMakeLists.txt
    ├── mobilenet_test.cc
    ├── fluid_generate_calib_test.cc
    ├── fluid_int8_test.cc
    ├── mobilenetv1
    │   ├── model
    │   └── params
    ├── run.sh
    └── run_impl.sh
    ```

    - `mobilenet_test.cc` is the c++ source code of inference using Paddle-TRT
    - `fluid_generate_calib_test.cc` is the c++ source code of inference using Paddle-TRT int8 calibration to generate calibration table
    - `fluid_int8_test.cc` is the c++ source code of inference using Paddle-TRT int8
    - `mobilenetv1` is the model dir
    - `run.sh` is the script for running inference

    Here we assume that the current directory is `SAMPLE_BASE_DIR/sample/paddle-TRT`.

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

## <a name="Paddle-TRT INT8 usage">Paddle-TRT INT8 usage</a>

1. Paddle-TRT INT8 introduction
    The parameters of the neural network are redundant to some extent. In many tasks, we can turn the Float32 model into Int8 model on the premise of precision. At present, Paddle-TRT supports to turn the trained Float32 model into Int8 model off line. The specific processes are as follows:

    1）**Create the calibration table**. We prepare about 500 real input data, and input the data to the model. Paddle-TRT will count the range information of each op input and output value in the model, and record in the calibration table. The information can reduce the information loss during model transformation.

    2）After creating the calibration table, run the model again, **Paddle-TRT will load the calibration table automatically**, and conduct the inference in the INT8 mode.

2. compile and test the INT8 example

    change the `mobilenet_test` in `run.sh` to `fluid_generate_calib_test` and run

    ``` shell
    sh run.sh
    ```

    We generate 500 input data to simulate the process, and it's suggested that you use real example for experiment. After the running period, there will be a new file named trt_calib_* under the `SAMPLE_BASE_DIR/sample/paddle-TRT/build/mobilenetv1/_opt_cache` model directory, which is the calibration table.

    Then copy the model dir with calibration infomation to path

    ``` shell
    cp -rf SAMPLE_BASE_DIR/sample/paddle-TRT/build/mobilenetv1/ SAMPLE_BASE_DIR/sample/paddle-TRT/mobilenetv1_calib
    ```

    change `fluid_generate_calib_test` in `run.sh` to `fluid_int8_test`, and change model dir path to `SAMPLE_BASE_DIR/sample/paddle-TRT/mobilenetv1_calib` and run

    ``` shell
    sh run.sh
    ```

## <a name="Paddle-TRT subgraph operation principle">Paddle-TRT subgraph operation principle</a>

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

## <a name="Paddle-TRT benchmark">Paddle-TRT benchmark</a>

### Test Environment
- CPU:Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz GPU:Tesla P4
- TensorRT 4.0, CUDA 8.0, CUDNN V7
- models: ResNet50，MobileNet，ResNet101, Inception V3.

### Test set
**PaddlePaddle, PyTorch, TensorFlow**

- PaddlePaddle integrates TensorRT with subgraph, model[link](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)。
- PyTorch uses original kernels, model[link1](https://github.com/pytorch/vision/tree/master/torchvision/models), [link2](https://github.com/marvis/pytorch-mobilenet)。
- We tested TF original and TF-TRT**对 TF—TRT 的测试并没有达到预期的效果，后期会对其进行补充**, model[link](https://github.com/tensorflow/models)。


#### ResNet50

|batch_size|PaddlePaddle(ms)|PyTorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1|4.64117 |16.3|10.878|
|5|6.90622| 22.9 |20.62|
|10|7.9758 |40.6|34.36|

#### MobileNet

|batch_size|PaddlePaddle(ms)|PyTorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1| 1.7541 | 7.8 |2.72|
|5| 3.04666 | 7.8 |3.19|
|10|4.19478 | 14.47 |4.25|

#### ResNet101

|batch_size|PaddlePaddle(ms)|PyTorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1|8.95767| 22.48 |18.78|
|5|12.9811 | 33.88 |34.84|
|10|14.1463| 61.97 |57.94|


#### Inception v3

|batch_size|PaddlePaddle(ms)|PyTorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1|15.1613 | 24.2 |19.1|
|5|18.5373 | 34.8 |27.2|
|10|19.2781| 54.8 |36.7|
