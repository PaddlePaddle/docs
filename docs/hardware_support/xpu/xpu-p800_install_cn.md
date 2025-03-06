# 昆仑芯 XPU P800 安装说明

飞桨框架 XPU 版支持昆仑芯 XPU P800 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译安装得到 wheel 包

## 昆仑芯 XPU P800 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 昆仑芯 P800 |
| 操作系统 | Ubuntu |

## 运行环境准备

推荐使用飞桨官方发布的昆仑芯 XPU 开发镜像，该镜像预装有昆仑芯基础运行环境库（XRE）。

```bash
# 拉取镜像
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310

# 参考如下命令，启动容器
docker run -it --name paddle-xpu-dev -v $(pwd):/work \
  -v /usr/local/bin/xpu-smi:/usr/local/bin/xpu-smi \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 /bin/bash

# 检查容器内是否可以正常识别昆仑芯 XPU 设备
xpu-smi
```

## 安装飞桨框架

**注意**：当前飞桨 develop 分支仅支持 X86 架构，如需昆仑芯 XPU 的 ARM 架构支持，请提交[issue](https://github.com/PaddlePaddle/Paddle/issues)告知我们

### 安装方式一：wheel 包安装

在启动的 docker 容器中，下载并安装飞桨官网发布的 wheel 包。

```bash
# 下载并安装 wheel 包
python -m pip install --pre paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
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
cmake .. -DPY_VERSION=3.10 -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=OFF -DWITH_XPU=ON -DON_INFER=ON \
    -DWITH_PYTHON=ON -DWITH_MKL=OFF -DWITH_XPU_BKCL=ON -DWITH_TESTING=ON -DWITH_DISTRIBUTE=ON -DWITH_XPU_XRE5=ON -DWITH_XCCL_RDMA=ON

# make 编译命令
make -j50 TARGET=HASWELL

# 编译产出在 build/python/dist/ 路径下，使用 pip 安装即可
pip install -U paddlepaddle_xpu-0.0.0-cp310-cp310-linux_x86_64.whl
```

## 基础功能检查

安装完成后，在 docker 容器中输入如下命令进行飞桨基础健康功能的检查。

```bash
# 检查当前安装版本
python -c "import paddle; paddle.version.show()"
# 预期得到输出如下
commit: 606d18c011a706c41b08b595821bbb835c44d637
cuda: False
cudnn: False
nccl: 0
xpu_xre: 5.0.21.15
xpu_xccl: 3.0.2.3
xpu_xhpc: dev/20250220
cinn: False
tensorrt: None
cuda_archs: []

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
