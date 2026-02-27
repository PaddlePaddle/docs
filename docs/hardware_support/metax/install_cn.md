
# 沐曦 曦云 C 系列 安装说明

飞桨框架 MACA 版支持基于沐曦 MACA 软件栈 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译安装得到 wheel 包

## 沐曦 曦云 C 系列 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 沐曦曦云 C 系列芯片，包括 C500 |
| 操作系统 | Linux 操作系统，包括 CentOS、Ubuntu、KylinV10 等 |

## 拉取镜像

推荐使用沐曦开发镜像。
```bash
# 拉取镜像
docker pull cr.metax-tech.com/public-ai-release/maca/paddle-metax:3.3.0-maca.ai3.3.0.10-py310-ubuntu22.04-amd64
```

```bash
# 考如下命令启动容器，ASCEND_RT_VISIBLE_DEVICES 可指定可见的 NPU 卡号
docker run
 -it --device=/dev/dri --device=/dev/mxcd --device=/dev/infiniband --group-add video \
 --name <container_name> --network=host --uts=host --ipc=host --privileged=true \
 --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
 --shm-size '500gb' --ulimit memlock=-1 -v /sw_home/:/sw_home/  -v /pde_ai/:/pde_ai/ \
 -v /mxstorage/:/mxstorage/ mxcr.io/ai-release/maca/paddle-metax:3.3.0-maca.ai3.3.0.0-py310-ubuntu22.04-amd64 /bin/bash
```

## 安装飞桨框架

### 安装方式一：wheel 包安装
沐曦曦云 C500 支持插件式安装，需先安装飞桨 CPU 安装包，再安装飞桨 沐曦 插件包：
```bash
# 先安装飞桨 CPU 安装包
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# 再安装飞桨 曦云 C500 插件包
python -m pip install paddle-metax-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/maca/
```

### 安装方式二：wheel 包（nightly）安装
沐曦曦云 C500 支持插件式安装，需先安装飞桨 CPU 安装包，再安装飞桨 沐曦 插件包：
```bash
# 先安装飞桨 CPU 安装包
python -m pip install  --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
# 再安装飞桨 曦云 C500 插件包
python -m pip install --pre paddle-metax-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/maca/
```
⚠️ 注意：nightly 版本为每日构建，可能存在不稳定性。如无新合入特性需求，建议使用 方式一 3.3 版本。

### 安装方式三：源代码编译安装
在启动的 docker 容器中，先安装飞桨 CPU 安装包，再下载 PaddleCustomDevice 源码编译得到飞桨 C500 插件包。

```bash
# 下载 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice.git

# 在 PaddleCUstomDevice 根目录下执行以下指令更新子模块代码
git submodule sync
git submodule update --init --recursive

# 进入硬件后端(沐曦 曦云 C500)目录
cd backends/metax_gpu

# 先安装飞桨 CPU 安装包
python -m pip install  --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# 编译安装
bash build_in_metax.sh
# 或者
bash change_patch.sh #只执行一次
bash compile.sh      # 可执行多次

# 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/*.whl --force-reinstall

```
## 如何卸载

请使用以下命令卸载 Paddle:

```bash
pip uninstall paddlepaddle paddle-metax-gpu
```
