# **兆芯下从源码编译**

## 环境准备

* **处理器：ZHAOXIN KaiSheng KH-40000，KH-50000；KaiXian KX-7000**
* **操作系统：麒麟 v10/v11，UOS20/25**
* **Python 版本 3.9/3.10/3.11/3.12/3.13 (64 bit)**
* **pip 或 pip3 版本 20.2.3+ (64 bit)**

兆芯平台为通用 x86 架构，支持 Windows 和 Linux 操作系统。在 Windows 下支持[PIP 安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/windows-pip.html)和[源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/windows-compile.html)，在 Linux 下支持[PIP 安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/linux-pip.html)、[Docker 安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/linux-docker.html)和[源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile-by-make.html)。
本文在兆芯平台上在国产化操作系统（麒麟/ UOS）中从源码编译并安装 PaddlePaddle，编译方法与[Linux 下从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile-by-make.html)基本一致。

## 编译步骤

ZHAOXIN 处理器为通用 x86 处理器，在国产化操作系统中支持[本机编译](#compile_from_docker)和[Docker 编译](#compile_from_host)，推荐使用[Docker 编译](#compile_from_host)；支持编译 CPU 版本和 GPU 版本。接下来详细介绍各个步骤。

### 克隆 PaddlePaddle 源代码

进入准备存储源代码的路径，用以下 Git 命令将 PaddlePaddle 源码从 GitHub 克隆到本地。
```
git clone --recursive https://github.com/PaddlePaddle/Paddle.git
```

### 安装 GPU 驱动

若要编译、安装、使用 Paddle GPU 版本，需要在系统内装好 GPU 驱动，例如：[NVidia 驱动](https://www.nvidia.cn/geforce/drivers/)
若要本地编译 Paddle GPU 版本，也可以随[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)安装 GPU 驱动。

<a name="ct_docker"></a>
### <span id="compile_from_docker">使用 Docker 编译</span>

#### 1. 安装 Docker：

国产化操作系统麒麟/ UOS 在其软件源中均提供了 Docker，可用自带的包管理器（apt 或 yum/dnf）来安装。虽然其版本略旧，但完全足够加载并运行[Docker 列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/docker/docker_list.html)中的镜像，编译 Paddle CPU 版本。
若要编译、安装、使用 Paddle GPU 版本，则需要 19.0.3 以上版本的 Docker，可参阅[教程](https://mirrors.tuna.tsinghua.edu.cn/help/docker-ce/)安装 docker-ce。

#### 2. 安装 GPU 容器套件：

若要编译、安装、使用 Paddle GPU 版本，则需安装 GPU 容器套件，用于在容器中使用 GPU，例如：[NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### 3. 拉取 PaddlePaddle 镜像：

* CPU 版本：
    ```
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:latest-dev
    ```
* GPU 版本：
    ```
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:cuda129-dev
    ```
`cuda129-dev` 仅为示例，百度提供多个 CUDA 版本对应的镜像，请根据自身需求选用。

#### 4. 创建并进入 Docker 容器：

进入源代码目录，运行如下命令创建容器，并将源代码目录挂载到容器的 /paddle 目录，最后登入容器：
* CPU 版本
    ```
    docker run --name paddle-cpu -v $PWD:/paddle --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
    ```
* GPU 版本
    ```
    docker run --gpus all --name paddle-gpu -v $PWD:/paddle --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:cuda129-dev /bin/bash
    ```

#### 5. 进入 Docker 后进入 paddle 目录：

```
cd /paddle
```

#### 6. 切换到 develop 分支：

```
git checkout develop
```

#### 7. 创建并进入 /paddle/build 路径：

```
mkdir -p /paddle/build && cd /paddle/build
```

#### 8. 安装 python 依赖：

```
pip3.10 install -r /paddle/python/requirements.txt
```
注意选择合适的 Python 版本，当前 Paddle 版本支持 Python 3.9/3.10/3.11/3.12/3.13，这些版本容器内都有。

#### 9. 执行 cmake

* CPU 版本：
    ```
    cmake .. -DPY_VERSION=3.10 -DWITH_GPU=OFF
    ```
* GPU 版本
    ```
    cmake .. -DPY_VERSION=3.10 -DWITH_GPU=ON -DWITH_DISTRIBUTE=ON
    ```
注意参数 `-DPY_VERSION` 指定的 Python 版本要跟前面执行的 pip 命令的版本一致。

#### 10. 执行编译：

```
make -j$(nproc)
```

<a name="ct_host"></a>
### <span id="compile_from_host">本机编译</span>

#### 1. 准备基础软件环境：

本机编译依赖本机安装的软件环境，最关键的有 Python 版本>=3.9，Gcc 版本>=8.2，Cmake 版本>=3.18。
如 OS 提供的源包含了合适版本的软件，直接通过包管理器安装即可。
如 OS 提供的源不包含合适版本的软件，则建议[使用 Docker 编译](#compile_from_docker)。
如坚持不使用 Docker，那么下面是几个解决基础软件版本问题的建议：
1. 可以[安装 uv](https://docs.astral.sh/uv/#installation)并用它来[创建新版本虚拟 Python 环境](https://docs.astral.sh/uv/pip/environments/#creating-a-virtual-environment)来解决 Python 的版本问题。激活 Python 虚拟环境后还需要[设置一些环境变量](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile-by-make.html#virtualenv-paddle-venv)。
2. 可以[从源码构建新版本的 cmake](https://github.com/Kitware/CMake)以解决其版本问题。
3. 可以[从源码构建新版本的 gcc](https://gcc.gnu.org/install/index.html)以解决其版本问题。

#### 2. 安装 GPU 版本所需的库：

编译 GPU 版本需要安装好所需的库，例如：[NVidia CUDA](https://developer.nvidia.com/cuda-toolkit-archive)运行时库、[NVidia cuDNN](https://developer.nvidia.com/cudnn)深度神经网络库、用于多 GPU 的[NVidia NCCL](https://developer.nvidia.com/nccl)通信库。

#### 3. 安装 patchelf：

Paddle 内部使用 patchelf 来修改动态库的 rpath，如果 OS 提供的源包括了 patchelf，直接安装即可，否则需要源码安装，请参考[patchelf 官方文档](https://github.com/NixOS/patchelf)。

#### 4. 进入 paddle 源代码的根目录：

```
cd <paddle 源代码所在路径>
```

#### 5. 切换到 develop 分支：

```
git checkout develop
```

#### 6. 安装 Python 依赖库：

```
pip install -r python/requirements.txt
```
实践中可能还需要安装 wheel 和 setuptools 两个 Python 库。

#### 7. 创建并进入 build 目录：

```
mkdir build && cd build
```

#### 8. 链接过程中打开文件数较多，可能超过系统默认限制导致编译出错，设置进程允许打开的最大文件数：

```
ulimit -n 4096
```

#### 9. 执行 cmake：

* CPU 版本：
    ```
    cmake .. -DPY_VERSION=3.10 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
    -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF
    ```
* GPU 版本：
    ```
    cmake .. -DPY_VERSION=3.10 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
    -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=ON -DWITH_DISTRIBUTE=ON
    ```
注意将 PY_VERSION 参数配置为实际的 python 版本。

#### 10. 编译：

```
make -j$(nproc)
```

## 安装

编译成功后进入`Paddle/build/python/dist`目录下找到生成的`.whl`包。
在当前机器或目标机器安装编译好的`.whl`包：
```
python3 -m pip install -U（whl 包的名字）
```
恭喜，至此您已完成 PaddlePaddle 在兆芯环境下的编译安装。

## 验证安装

安装完成后您可以使用 `python` 或 `python3` 进入 python 解释器，输入
```
import paddle
```
再输入
```
paddle.utils.run_check()
```
如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

在 mobilenetv1 和 resnet50 模型上测试
```
wget -O profile.tar https://paddle-cetc15.bj.bcebos.com/profile.tar?authorization=bce-auth-v1/4409a3f3dd76482ab77af112631f01e4/2020-10-09T10:11:53Z/-1/host/786789f3445f498c6a1fd4d9cd3897ac7233700df0c6ae2fd78079eba89bf3fb
```
```
tar xf profile.tar && cd profile
```
```
python resnet.py --model_file ResNet50_inference/model --params_file ResNet50_inference/params
# 正确输出应为：[0.0002414  0.00022418 0.00053661 0.00028639 0.00072682 0.000213
#              0.00638718 0.00128127 0.00013535 0.0007676 ]
```
```
python mobilenetv1.py --model_file mobilenetv1/model --params_file mobilenetv1/params
# 正确输出应为：[0.00123949 0.00100392 0.00109539 0.00112206 0.00101901 0.00088412
#              0.00121536 0.00107679 0.00106071 0.00099605]
```
```
python ernie.py --model_dir ernieL3H128_model/
# 正确输出应为：[0.49879393 0.5012061 ]
```

## 卸载

请使用以下命令卸载 PaddlePaddle：
```
python3 -m pip uninstall paddlepaddle
```

## 备注

已在 ZHAOXIN 下测试过 resnet50，mobilenetv1，ernie，ELMo 等模型，基本保证了预测使用算子的正确性，如果您在使用过程中遇到计算结果错误，编译失败等问题，请到[issue](https://github.com/PaddlePaddle/Paddle/issues)中留言，我们会及时解决。

预测文档见[doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html)，使用示例见[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)
