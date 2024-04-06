# 海光 DCU 安装说明

飞桨框架 DCU 版支持海光 DCU 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译安装得到 wheel 包

## 海光 DCU 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 海光 Z100 系列芯片，包括 Z100、Z100L |
| 操作系统 | Linux 操作系统，包括 CentOS、KylinV10 |

## 运行环境准备

推荐使用飞桨官方发布的海光 DCU 开发镜像，该镜像预装有海光 DCU 基础运行环境库（DTK）。

```bash
# 拉取镜像
docker pull registry.baidubce.com/device/paddle-dcu:dtk23.10.1-kylinv10-gcc73-py310

# 参考如下命令，启动容器
docker run -it --name paddle-dcu-dev -v $(pwd):/work \
  --workdir=/work --shm-size=128G --network=host  \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  registry.baidubce.com/device/paddle-dcu:dtk23.10.1-kylinv10-gcc73-py310 /bin/bash

# 检查容器内是否可以正常识别海光 DCU 设备
rocm-smi

# 预期得到输出如下
============System Management Interface ============
====================================================
DCU  Temp   AvgPwr  Fan   Perf  PwrCap  VRAM%  DCU%
0    30.0c  38.0W   0.0%  auto  280.0W    0%   0%
1    30.0c  41.0W   0.0%  auto  280.0W    0%   0%
2    29.0c  38.0W   0.0%  auto  280.0W    0%   0%
3    29.0c  39.0W   0.0%  auto  280.0W    0%   0%
====================================================
===================End of SMI Log===================
```

## 安装飞桨框架

**注意**：飞桨框架 DCU 版仅支持海光 C86 架构。

### 安装方式一：wheel 包安装

在启动的 docker 容器中，下载并安装飞桨官网发布的 wheel 包。

```bash
# 下载 wheel 包
wget https://paddle-device.bj.bcebos.com/0.0.0/dcu/paddlepaddle_rocm-0.0.0-cp310-cp310-linux_x86_64.whl

# 安装 wheel 包
pip install -U paddlepaddle_rocm-0.0.0-cp310-cp310-linux_x86_64.whl
```

### 安装方式二：源代码编译安装

在启动的 docker 容器中，下载 Paddle 源码并编译，CMAKE 编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)。

```bash
# 下载 Paddle 源码
git clone https://github.com/PaddlePaddle/Paddle.git -b develop
cd Paddle

# 创建编译目录
mkdir build && cd build

# cmake 编译命令
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS="-Wno-error -w" \
  -DPY_VERSION=3.10 -DPYTHON_EXECUTABLE=`which python3` -DWITH_CUSTOM_DEVICE=OFF \
  -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_MKL=ON \
  -DWITH_ROCM=ON -DWITH_RCCL=ON

# make 编译命令
make -j16

# 编译产出在 build/python/dist/ 路径下，使用 pip 安装即可
pip install -U paddlepaddle_rocm-0.0.0-cp310-cp310-linux_x86_64.whl
```

## 基础功能检查

安装完成后，在 docker 容器中输入如下命令进行飞桨基础健康功能的检查。

```bash
# 检查当前安装版本
python -c "import paddle; paddle.version.show()"
# 预期得到输出如下
commit: d37bd8bcf75cf51f6c1117526f3f67d04946ebb9
cuda: False
cudnn: False
nccl: 0

# 飞桨基础健康检查
python -c "import paddle; paddle.utils.run_check()"
# 预期得到输出如下
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 GPU.
PaddlePaddle works well on 8 GPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 如何卸载

请使用以下命令卸载：

```bash
pip uninstall paddlepaddle-rocm
```
