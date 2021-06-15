# 使用Paddle-TensorRT库预测

NVIDIA TensorRT 是一个高性能的深度学习预测库，可为深度学习推理应用程序提供低延迟和高吞吐量。PaddlePaddle 采用子图的形式对TensorRT进行了集成，即我们可以使用该模块来提升Paddle模型的预测性能。该模块依旧在持续开发中，目前支持的模型如下表所示：

|分类模型|检测模型|分割模型|
|---|---|---|
|mobilenetv1|yolov3|ICNET|
|resnet50|SSD||
|vgg16|mask-rcnn||
|resnext|faster-rcnn||
|AlexNet|cascade-rcnn||
|Se-ResNext|retinanet||
|GoogLeNet|mobilenet-SSD||
|DPN|||

在这篇文档中，我们将会对Paddle-TensorRT库的获取、使用和原理进行介绍。

**Note:**

1. 从源码编译时，TensorRT预测库目前仅支持使用GPU编译，且需要设置编译选项TENSORRT_ROOT为TensorRT所在的路径。
2. Windows支持需要TensorRT 版本5.0以上。
3. Paddle-TRT目前仅支持固定输入shape。
4. 下载安装TensorRT后，需要手动在`NvInfer.h`文件中为`class IPluginFactory`和`class IGpuAllocator`分别添加虚析构函数：
    ``` c++
    virtual ~IPluginFactory() {};
    virtual ~IGpuAllocator() {};
    ```

## 内容
- [Paddle-TRT使用介绍](#Paddle-TRT使用介绍)
- [Paddle-TRT样例编译测试](#Paddle-TRT样例编译测试)
- [Paddle-TRT INT8使用](#Paddle-TRT_INT8使用)
- [Paddle-TRT子图运行原理](#Paddle-TRT子图运行原理)
- [Paddle-TRT性能测试](#Paddle-TRT性能测试)

## <a name="Paddle-TRT使用介绍">Paddle-TRT使用介绍</a>

在使用AnalysisPredictor时，我们通过配置AnalysisConfig中的接口

``` c++
config->EnableTensorRtEngine(1 << 20      /* workspace_size*/,  
                        batch_size        /* max_batch_size*/,  
                        3                 /* min_subgraph_size*/,
                        AnalysisConfig::Precision::kFloat32 /* precision*/,
                        false             /* use_static*/,
                        false             /* use_calib_mode*/);
```  
的方式来指定使用Paddle-TRT子图方式来运行。
该接口中的参数的详细介绍如下：

- **`workspace_size`**，类型：int，默认值为1 << 20。指定TensorRT使用的工作空间大小，TensorRT会在该大小限制下筛选合适的kernel执行预测运算。
- **`max_batch_size`**，类型：int，默认值为1。需要提前设置最大的batch大小，运行时batch大小不得超过此限定值。
- **`min_subgraph_size`**，类型：int，默认值为3。Paddle-TRT是以子图的形式运行，为了避免性能损失，当子图内部节点个数大于`min_subgraph_size`的时候，才会使用Paddle-TRT运行。
- **`precision`**，类型：`enum class Precision {kFloat32 = 0, kHalf, kInt8,};`, 默认值为`AnalysisConfig::Precision::kFloat32`。指定使用TRT的精度，支持FP32（kFloat32），FP16（kHalf），Int8（kInt8）。若需要使用Paddle-TRT int8离线量化校准，需设定`precision`为 `AnalysisConfig::Precision::kInt8`, 且设置`use_calib_mode` 为true。
- **`use_static`**，类型：bool, 默认值为false。如果指定为true，在初次运行程序的时候会将TRT的优化信息进行序列化到磁盘上，下次运行时直接加载优化的序列化信息而不需要重新生成。
- **`use_calib_mode`**，类型：bool, 默认值为false。若要运行Paddle-TRT int8离线量化校准，需要将此选项设置为true。

**Note：** Paddle-TRT目前只支持固定shape的输入，不支持变化shape的输入。

## <a name="Paddle-TRT样例编译测试">Paddle-TRT样例编译测试</a>

1. 下载或编译带有 TensorRT 的paddle预测库，参考[安装与编译C++预测库](../../inference_deployment/inference/build_and_install_lib_cn.html)。
2. 从[NVIDIA官网](https://developer.nvidia.com/nvidia-tensorrt-download)下载对应本地环境中cuda和cudnn版本的TensorRT，需要登陆NVIDIA开发者账号。
3. 下载[预测样例](https://paddle-inference-dist.bj.bcebos.com/tensorrt_test/paddle_inference_sample_v1.7.tar.gz)并解压，进入`sample/paddle-TRT`目录下。

    `paddle-TRT` 文件夹目录结构如下：

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

    - `mobilenet_test.cc` 为使用paddle-TRT预测的C++源文件
    - `fluid_generate_calib_test.cc` 为使用TRT int8离线量化校准的C++源文件
    - `fluid_int8_test.cc` 为使用TRT执行int8预测的C++源文件
    - `mobilenetv1` 为模型文件夹
    - `run.sh` 为预测运行脚本文件

    在这里假设样例所在的目录为 `SAMPLE_BASE_DIR/sample/paddle-TRT`

4. 配置编译与运行脚本

    编译运行预测样例之前，需要根据运行环境配置编译与运行脚本`run.sh`。`run.sh`的选项与路径配置的部分如下：

    ```shell
    # 设置是否开启MKL、GPU、TensorRT，如果要使用TensorRT，必须打开GPU
    WITH_MKL=ON
    WITH_GPU=ON
    USE_TENSORRT=ON

    # 按照运行环境设置预测库路径、CUDA库路径、CUDNN库路径、TensorRT路径、模型路径
    LIB_DIR=YOUR_LIB_DIR
    CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
    CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
    TENSORRT_ROOT_DIR=YOUR_TENSORRT_ROOT_DIR
    MODEL_DIR=YOUR_MODEL_DIR
    ```

    按照实际运行环境配置`run.sh`中的选项开关和所需lib路径。

5. 编译与运行样例  


## <a name="Paddle-TRT_INT8使用">Paddle-TRT INT8使用</a>

1. Paddle-TRT INT8 简介  
    神经网络的参数在一定程度上是冗余的，在很多任务上，我们可以在保证模型精度的前提下，将Float32的模型转换成Int8的模型。目前，Paddle-TRT支持离线将预训练好的Float32模型转换成Int8的模型，具体的流程如下：

    1) **生成校准表**（Calibration table）：我们准备500张左右的真实输入数据，并将数据输入到模型中去，Paddle-TRT会统计模型中每个op输入和输出值的范围信息，并将其记录到校准表中，这些信息有效减少了模型转换时的信息损失。

    2) 生成校准表后，再次运行模型，**Paddle-TRT会自动加载校准表**，并进行INT8模式下的预测。

2. 编译测试INT8样例
    将`run.sh`文件中的`mobilenet_test`改为`fluid_generate_calib_test`，运行

    ``` shell  
    sh run.sh  
    ```

    即可执行生成校准表样例，在该样例中，我们随机生成了500个输入来模拟这一过程，在实际业务中，建议大家使用真实样例。运行结束后，在 `SAMPLE_BASE_DIR/sample/paddle-TRT/build/mobilenetv1/_opt_cache` 模型目录下会多出一个名字为trt_calib_*的文件，即校准表。

    生成校准表后，将带校准表的模型文件拷贝到特定地址

    ``` shell  
    cp -rf SAMPLE_BASE_DIR/sample/paddle-TRT/build/mobilenetv1/ SAMPLE_BASE_DIR/sample/paddle-TRT/mobilenetv1_calib  
    ```

    将`run.sh`文件中的`fluid_generate_calib_test`改为`fluid_int8_test`，将模型路径改为`SAMPLE_BASE_DIR/sample/paddle-TRT/mobilenetv1_calib`，运行

    ``` shell  
    sh run.sh  
    ```

    即可执行int8预测样例。

## <a name="Paddle-TRT子图运行原理">Paddle-TRT子图运行原理</a>

   PaddlePaddle采用子图的形式对TensorRT进行集成，当模型加载后，神经网络可以表示为由变量和运算节点组成的计算图。Paddle TensorRT实现的功能是对整个图进行扫描，发现图中可以使用TensorRT优化的子图，并使用TensorRT节点替换它们。在模型的推断期间，如果遇到TensorRT节点，Paddle会调用TensorRT库对该节点进行优化，其他的节点调用Paddle的原生实现。TensorRT在推断期间能够进行Op的横向和纵向融合，过滤掉冗余的Op，并对特定平台下的特定的Op选择合适的kernel等进行优化，能够加快模型的预测速度。  

下图使用一个简单的模型展示了这个过程：  

**原始网络**
<p align="center">
 <img src="https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_original.png" width="600">
</p>

**转换的网络**
<p align="center">
 <img src="https://raw.githubusercontent.com/NHZlX/FluidDoc/add_trt_doc/doc/fluid/user_guides/howto/inference/image/model_graph_trt.png" width="600">
</p>


   我们可以在原始模型网络中看到，绿色节点表示可以被TensorRT支持的节点，红色节点表示网络中的变量，黄色表示Paddle只能被Paddle原生实现执行的节点。那些在原始网络中的绿色节点被提取出来汇集成子图，并由一个TensorRT节点代替，成为转换后网络中的`block-25` 节点。在网络运行过程中，如果遇到该节点，Paddle将调用TensorRT库来对其执行。

## <a name="Paddle-TRT性能测试">Paddle-TRT性能测试</a>

### 测试环境
- CPU:Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz GPU:Tesla P4
- TensorRT4.0, CUDA8.0, CUDNNV7
- 测试模型 ResNet50，MobileNet，ResNet101, Inception V3.

### 测试对象
**PaddlePaddle, Pytorch, Tensorflow**

- 在测试中，PaddlePaddle使用子图优化的方式集成了TensorRT, 模型[地址](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification/models)。
- Pytorch使用了原生的实现, 模型[地址1](https://github.com/pytorch/vision/tree/master/torchvision/models)、[地址2](https://github.com/marvis/pytorch-mobilenet)。
- 对TensorFlow测试包括了对TF的原生的测试，和对TF—TRT的测试，**对TF—TRT的测试并没有达到预期的效果，后期会对其进行补充**， 模型[地址](https://github.com/tensorflow/models)。


#### ResNet50

|batch_size|PaddlePaddle(ms)|Pytorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1|4.64117 |16.3|10.878|
|5|6.90622| 22.9 |20.62|
|10|7.9758 |40.6|34.36|

#### MobileNet

|batch_size|PaddlePaddle(ms)|Pytorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1| 1.7541 | 7.8 |2.72|
|5| 3.04666 | 7.8 |3.19|
|10|4.19478 | 14.47 |4.25|

#### ResNet101

|batch_size|PaddlePaddle(ms)|Pytorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1|8.95767| 22.48 |18.78|
|5|12.9811 | 33.88 |34.84|
|10|14.1463| 61.97 |57.94|


#### Inception v3

|batch_size|PaddlePaddle(ms)|Pytorch(ms)|TensorFlow(ms)|
|---|---|---|---|
|1|15.1613 | 24.2 |19.1|
|5|18.5373 | 34.8 |27.2|
|10|19.2781| 54.8 |36.7|
