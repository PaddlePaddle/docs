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

您可以基于 docker、pip、源码等不同方式准备飞桨开发环境

### 基于 Docker 的方式（推荐）

我们推荐使用飞桨官方发布的海光 DCU 开发镜像，该镜像预装有海光 DCU 基础运行环境库（DTK）和飞桨 3.0rc 版本的 SDK。

```bash
# 拉取镜像
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-dcu:3.0.0rc1-dtk24.04.1-kylinv10-gcc82-py310
```

```bash
# 启动容器
docker run -it --name paddle-dcu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-dcu:3.0.0rc1-dtk24.04.1-kylinv10-gcc82-py310 /bin/bash
```

#### 选项说明及可调整参数

##### ① `--name paddle-dcu-dev`
- **作用**：指定容器名称。
- **可调整**：
  - 用户可改为其他名称，例如 `paddle-dcu-test`，方便区分不同实验。

##### ② `-v $(pwd):/work`
- **作用**：挂载本地目录到容器内 `/work` 目录。
- **可调整**：
  - 可以修改 `$(pwd)` 为实际路径，例如 `-v /data/projects:/work`，让容器访问宿主机的数据。

##### ③ `--shm-size=128G`
- **作用**：设置共享内存大小，影响数据处理和计算效率。
- **可调整**：
  - 若内存有限，可降低，如 `--shm-size=32G`，但可能影响大规模训练。
  - 若训练任务需要更大共享内存，可提高，如 `--shm-size=256G`。

```bash
# 检查容器内是否正常识别海光 DCU 设备
rocm-smi
```

```bash
# 预期输出
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

### 基于 pip 安装的方式

```bash
# 下载并安装 wheel 包
python -m pip install paddlepaddle-dcu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
```

### 基于源码编译的方式

```bash
# 下载 Paddle 源码
git clone https://github.com/PaddlePaddle/Paddle.git -b release/3.0-rc
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
python -m pip install -U paddlepaddle_dcu-*-linux_x86_64.whl
```

## 基础功能检查

输入如下命令进行飞桨基础健康功能的检查。

```bash
# 检查当前安装版本
python -c "import paddle; paddle.version.show()"
```
```bash
# 预期得到输出如下
full_version: 3.0.0-rc1
major: 3
minor: 0
patch: 0-rc1
rc: 0
cuda: False
cudnn: False
nccl: 0
xpu_xre: False
xpu_xccl: False
xpu_xhpc: False
cinn: False
tensorrt_version: None
cuda_archs: []
```
```bash
# 飞桨基础健康检查
python -c "import paddle; paddle.utils.run_check()"
```
```bash
# 预期得到输出如下
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 8 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
