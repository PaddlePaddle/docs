## 运行C++ YOLOv3图像检测样例

### 一：获取YOLOv3模型

点击[链接](https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/yolov3_infer.tar.gz)下载模型， 该模型在imagenet数据集训练得到的，如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleDetection)。

### 二：**样例编译**

文件`yolov3_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。  
文件`CMakeLists.txt` 为编译构建文件。  
脚本`run_impl.sh` 包含了第三方库、预编译库的信息配置。

编译yolov3样例，我们首先需要对脚本`run_impl.sh` 文件中的配置进行修改。

1）**修改`run_impl.sh`**

打开`run_impl.sh`，我们对以下的几处信息进行修改：

```shell
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

# 配置预测库的根目录
LIB_DIR=${YOUR_LIB_DIR}/fluid_inference_install_dir

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/local/cudnn/lib64
CUDA_LIB=/usr/local/cuda/lib64
# TENSORRT_ROOT=/usr/local/TensorRT-6.0.1.5
```

运行 `sh run_impl.sh`， 会在目录下产生build目录。


2） **运行样例**

```shell
# 进入build目录
cd build
# 运行样例
./yolov3_test -model_file ${YOLO_MODEL_PATH}/__model__ --params_file ${YOLO_MODEL_PATH}/__params__
```

运行结束后，程序会将模型输出个数打印到屏幕，说明运行成功。

### 更多链接
- [Paddle Inference使用Quick Start！]()
- [Paddle Inference Python Api使用]()
