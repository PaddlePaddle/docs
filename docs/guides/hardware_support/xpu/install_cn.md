# 昆仑 XPU 安装说明

飞桨框架 XPU 版支持昆仑芯 XPU 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译安装得到 wheel 包

## 昆仑 XPU 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 昆仑芯 2 代，包括 R200、R300、R200-8F、R200-8FS、RG800 |
| 操作系统 | Linux 操作系统，包括 Ubuntu、CentOS、KylinV10 |

## 运行环境准备

推荐使用飞桨官方发布的昆仑 XPU 开发镜像，该镜像预装有昆仑基础运行环境库（XRE）。

```bash
# 拉取镜像
docker pull registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310

# 参考如下命令，启动容器
docker run -it --name paddle-xpu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 /bin/bash

# 检查容器内是否可以正常识别昆仑 XPU 设备
xpu_smi

# 预期得到输出如下
Runtime Version: 4.31
Driver Version: 4.0
  DEVICES
-------------------------------------------------------------------------------------------
| DevID |   PCI Addr   | Model |   SN   |    INODE   | UseRate |    L3     |    Memory    |
-------------------------------------------------------------------------------------------
|     0 | 0000:53:00.0 | R300  | 02Kxxx | /dev/xpu0  |     0 % | 0 / 63 MB | 0 / 32768 MB |
|     1 | 0000:56:00.0 | R300  | 02Kxxx | /dev/xpu1  |     0 % | 0 / 63 MB | 0 / 32768 MB |
-------------------------------------------------------------------------------------------
  VIDEO
-----------------------------------------------------------------------------------
| DevID | Model |         DEC         |         ENC         |       IMGPROC       |
-----------------------------------------------------------------------------------
|     0 |  R300 | 0 %, 0 fps, 800 MHz | 0 %, 0 fps, 800 MHz | 0 %, 0 fps, 800 MHz |
|     1 |  R300 | 0 %, 0 fps, 800 MHz | 0 %, 0 fps, 800 MHz | 0 %, 0 fps, 800 MHz |
-----------------------------------------------------------------------------------
  PROCESSES
-------------------------------------------------
| DevID | PID | Streams | L3 | Memory | Command |
-------------------------------------------------
-------------------------------------------------
```

## 安装飞桨框架

**注意**：当前飞桨 develop 分支仅支持 X86 架构，如需昆仑 XPU 的 ARM 架构支持，请切换到 [release/2.6](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/guides/hardware_support/xpu/install_cn.html) 分支。

### 安装方式一：wheel 包安装

在启动的 docker 容器中，下载并安装飞桨官网发布的 wheel 包。

```bash
# 下载并安装 wheel 包
pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu
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
  -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_DISTRIBUTE=ON -DWITH_ARM=OFF \
  -DWITH_XPU=ON -DWITH_XPU_BKCL=ON -DWITH_UBUNTU=ON

# make 编译命令
make -j16

# 编译产出在 build/python/dist/ 路径下，使用 pip 安装即可
pip install -U paddlepaddle_xpu-0.0.0-cp310-cp310-linux_x86_64.whl
```

## 基础功能检查

安装完成后，在 docker 容器中输入如下命令进行飞桨基础健康功能的检查。

```bash
# 检查当前安装版本
python -c "import paddle; paddle.version.show()"
# 预期得到输出如下
commit: 84425362060e126b066a5a0f0d29ae2e2218a834
xpu: 20240104
xpu_xccl: 1.1.8.1
xpu_xhpc: 20240312

# 飞桨基础健康检查
python -c "import paddle; paddle.utils.run_check()"
# 预期得到输出如下
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 XPU.
PaddlePaddle works well on 8 XPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 如何卸载

请使用以下命令卸载：

```bash
pip uninstall paddlepaddle-xpu
```
