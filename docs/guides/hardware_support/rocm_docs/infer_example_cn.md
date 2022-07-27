# 飞桨框架 ROCm 版预测示例

使用海光 CPU/DCU 进行预测与使用 Intel CPU/Nvidia GPU 预测相同，支持飞桨原生推理库(Paddle Inference)，适用于高性能服务器端、云端推理。当前 Paddle ROCm 版本完全兼容 Paddle CUDA 版本的 C++/Python API，直接使用原有的 GPU 预测命令和参数即可。

## C++预测部署

**注意**：更多 C++预测 API 使用说明请参考 [Paddle Inference - C++ API](https://paddleinference.paddlepaddle.org.cn/api_reference/cxx_api_index.html)

**第一步**：源码编译 C++预测库

当前 Paddle ROCm 版只支持通过源码编译的方式提供 C++预测库。编译环境准备请参考 [飞桨框架 ROCm 版安装说明：通过源码编译安装](./paddle_install_cn.html)。

```bash
# 下载源码，切换到 release/2.1 分支
git clone -b release/2.1 https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行 cmake，注意这里需打开预测优化选项 ON_INFER
cmake .. -DPY_VERSION=3.7 -DWITH_ROCM=ON -DWITH_TESTING=OFF -DON_INFER=ON \
         -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 使用以下命令来编译
make -j$(nproc)
```

编译完成之后，`build` 目录下的 `paddle_inference_install_dir` 即为 C++ 预测库，目录结构如下：

```bash
build/paddle_inference_install_dir
├── CMakeCache.txt
├── paddle
│   ├── include                                    C++ 预测库头文件目录
│   │   ├── crypto
│   │   ├── experimental
│   │   ├── internal
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_infer_declare.h
│   │   ├── paddle_inference_api.h                 C++ 预测库头文件
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   ├── paddle_pass_builder.h
│   │   └── paddle_tensor.h
│   └── lib
│       ├── libpaddle_inference.a                  C++ 静态预测库文件
│       └── libpaddle_inference.so                 C++ 动态态预测库文件
├── third_party
│   ├── install                                    第三方链接库和头文件
│   │   ├── cryptopp
│   │   ├── gflags
│   │   ├── glog
│   │   ├── mkldnn
│   │   ├── mklml
│   │   ├── protobuf
│   │   └── xxhash
│   └── threadpool
│       └── ThreadPool.h
└── version.txt
```

其中 `version.txt` 文件中记录了该预测库的版本信息，包括 Git Commit ID、使用 OpenBlas 或 MKL 数学库、ROCm/MIOPEN 版本号，如：

```bash
GIT COMMIT ID: e75412099f97a49701324788b468d80391293ea9
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: OFF
WITH_ROCM: ON
HIP version: 4.0.20496-4f163c68
MIOpen version: v2.11
CXX compiler version: 7.3.1
```

**第二步**：准备预测部署模型

下载 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle 预测格式的模型，位于文件夹 ResNet50 下。如需查看模型结构，可将 inference.pdmodel 文件通过模型可视化工具 [Netron](https://netron.app/) 打开。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得模型目录即文件如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

**第三步**：获取预测示例代码并编译运行

**预先要求**：

本章节 C++ 预测示例代码位于 [Paddle-Inference-Demo/c++/resnet50](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++/resnet50)。

请先将示例代码下载到本地，再将第一步中编译得到的 `paddle_inference_install_dir` 重命名为 `paddle_inference` 文件夹，移动到示例代码的 `Paddle-Inference-Demo/c++/lib` 目录下。使用到的文件如下所示：

```bash
-rw-r--r-- 1 root root 3479 Jun  2 03:14 README.md                 README 说明
-rw-r--r-- 1 root root 3051 Jun  2 03:14 resnet50_test.cc          预测 C++ 源码程序
drwxr-xr-x 2 root root 4096 Mar  5 07:43 resnet50                  第二步中下载并解压的预测部署模型文件夹
-rw-r--r-- 1 root root  387 Jun  2 03:14 run.sh                    运行脚本
-rwxr-xr-x 1 root root 1077 Jun  2 03:14 compile.sh                编译脚本
-rw-r--r-- 1 root root 9032 Jun  2 07:26 ../lib/CMakeLists.txt     CMAKE 文件
drwxr-xr-x 1 root root 9032 Jun  2 07:26 ../lib/paddle_inference   第一步编译的到的 Paddle Infernece C++ 预测库文件夹
```

编译运行预测样例之前，需要根据运行环境配置编译脚本 `compile.sh`。

```bash
# 根据预编译库中的 version.txt 信息判断是否将以下标记打开
WITH_MKL=ON
WITH_GPU=OFF # 注意这里需要关掉 WITH_GPU
USE_TENSORRT=OFF

WITH_ROCM=ON # 注意这里需要打开 WITH_ROCM
ROCM_LIB=/opt/rocm/lib
```

运行 `run.sh` 脚本进行编译和运行，即可获取最后的预测结果：

```bash
bash run.sh

# 成功执行之后，得到的预测输出结果如下：
... ...
I0602 04:12:03.708333 52627 analysis_predictor.cc:595] ======= optimize end =======
I0602 04:12:03.709321 52627 naive_executor.cc:98] ---  skip [feed], feed -> inputs
I0602 04:12:03.710139 52627 naive_executor.cc:98] ---  skip [save_infer_model/scale_0.tmp_1], fetch -> fetch
I0602 04:12:03.711813 52627 device_context.cc:624] oneDNN v2.2.1
WARNING: Logging before InitGoogleLogging() is written to STDERR
I0602 04:12:04.106405 52627 resnet50_test.cc:73] run avg time is 394.801 ms
I0602 04:12:04.106503 52627 resnet50_test.cc:88] 0 : 0
I0602 04:12:04.106525 52627 resnet50_test.cc:88] 100 : 2.04163e-37
I0602 04:12:04.106552 52627 resnet50_test.cc:88] 200 : 2.1238e-33
I0602 04:12:04.106573 52627 resnet50_test.cc:88] 300 : 0
I0602 04:12:04.106591 52627 resnet50_test.cc:88] 400 : 1.6849e-35
I0602 04:12:04.106603 52627 resnet50_test.cc:88] 500 : 0
I0602 04:12:04.106618 52627 resnet50_test.cc:88] 600 : 1.05767e-19
I0602 04:12:04.106643 52627 resnet50_test.cc:88] 700 : 2.04094e-23
I0602 04:12:04.106670 52627 resnet50_test.cc:88] 800 : 3.85254e-25
I0602 04:12:04.106683 52627 resnet50_test.cc:88] 900 : 1.52391e-30
```

## Python 预测部署示例

**注意**：更多 Python 预测 API 使用说明请参考 [Paddle Inference - Python API](https://paddleinference.paddlepaddle.org.cn/api_reference/python_api_index.html)

**第一步**：安装 Python 预测库

Paddle ROCm 版的 Python 预测库请参考 [飞桨框架 ROCm 版安装说明](./paddle_install_cn.html) 进行安装或编译。

**第二步**：准备预测部署模型

下载 [ResNet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle 预测格式的模型，位于文件夹 ResNet50 下。如需查看模型结构，可将 inference.pdmodel 文件通过模型可视化工具 [Netron](https://netron.app/) 打开。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得模型目录即文件如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams.info
└── inference.pdiparams
```

**第三步**：准备预测部署程序

将以下代码保存为 `python_demo.py` 文件：

```bash
import argparse
import numpy as np

# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

def main():
    args = parse_args()

    # 创建 config
    config = paddle_infer.Config(args.model_file, args.params_file)

    # 根据 config 创建 predictor
    predictor = paddle_infer.create_predictor(config)

    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 设置输入
    fake_input = np.random.randn(args.batch_size, 3, 318, 318).astype("float32")
    input_handle.reshape([args.batch_size, 3, 318, 318])
    input_handle.copy_from_cpu(fake_input)

    # 运行 predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu() # numpy.ndarray 类型
    print("Output data size is {}".format(output_data.size))
    print("Output data shape is {}".format(output_data.shape))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename")
    parser.add_argument("--params_file", type=str, help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    return parser.parse_args()

if __name__ == "__main__":
    main()
```

**第四步**：执行预测程序

```bash
# 参数输入为本章节第 2 步中下载的 ResNet50 模型
python python_demo.py --model_file ./resnet50/inference.pdmodel \
                      --params_file ./resnet50/inference.pdiparams \
                      --batch_size 2

# 成功执行之后，得到的预测输出结果如下：
... ...
I0602 04:14:13.455812 52741 analysis_predictor.cc:595] ======= optimize end =======
I0602 04:14:13.456934 52741 naive_executor.cc:98] ---  skip [feed], feed -> inputs
I0602 04:14:13.458562 52741 naive_executor.cc:98] ---  skip [save_infer_model/scale_0.tmp_1], fetch -> fetch
Output data size is 2000
Output data shape is (2, 1000)
```
