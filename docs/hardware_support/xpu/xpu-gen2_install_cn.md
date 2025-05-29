# 昆仑芯 XPU 安装说明

飞桨框架 XPU 版支持昆仑芯 XPU 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译安装得到 wheel 包

## 昆仑芯 XPU 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 昆仑芯 2 代，包括 R200、R300、R200-8F、RG800 |
| 操作系统 | Linux 操作系统，包括 Ubuntu、CentOS、KylinV10 |

**注意**：当前教程适用于『昆仑芯』二代芯片。查看芯片类型请参考如下命令：

```bash
# 系统环境下运行如下命令，如果有设备列表输出，且字段为 3684，则说明芯片为昆仑芯二代芯片
lspci -d 1d22: -n
```

## 运行环境准备

推荐使用飞桨官方发布的昆仑芯 XPU 开发镜像，该镜像预装有昆仑芯基础运行环境库（XRE）。

```bash
# 拉取镜像
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
```
```bash
# 参考如下命令，启动容器
docker run -it --name paddle-xpu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 /bin/bash
```
#### 选项说明及可调整参数

##### ① `--name paddle-xpu-dev`
- **作用**：指定容器名称。
- **可调整**：
  - 用户可改为其他名称，例如 `paddle-xpu-test`，方便区分不同实验。

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
# 检查容器内是否可以正常识别昆仑芯 XPU 设备
xpu_smi
```
```bash
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

**注意**：当前飞桨 develop 分支仅支持 X86 架构，如需昆仑芯 XPU 的 ARM 架构支持，请切换到 [release/2.6](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.6/guides/hardware_support/xpu/install_cn.html) 分支。

### 安装方式一：wheel 包安装

在启动的 docker 容器中，下载并安装飞桨官网发布的 wheel 包。

```bash
# 下载并安装 wheel 包
pip install paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu
```
⚠️ 注意：nightly 版本为每日构建，可能存在不稳定性。如果需要更稳定的版本，建议使用 3.0-rc 版本。
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
⚠️ 注意：nightly 版本为每日构建，可能存在不稳定性。如果需要更稳定的版本，建议使用 3.0-rc 版本。
## 基础功能检查

安装完成后，在 docker 容器中输入如下命令进行飞桨基础健康功能的检查。

```bash
# 检查当前安装版本
python -c "import paddle; paddle.version.show()"
```
```bash
# 预期得到输出如下
commit: 84425362060e126b066a5a0f0d29ae2e2218a834
xpu: 20240104
xpu_xccl: 1.1.8.1
xpu_xhpc: 20240312
```
```bash
# 飞桨基础健康检查
python -c "import paddle; paddle.utils.run_check()"
```
```bash
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
