# 沐曦 曦云C系列 安装说明

飞桨框架 MACA 版支持基于沐曦 MACA软件栈 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译安装得到 wheel 包

## 沐曦 曦云C系列 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 沐曦曦云C 系列芯片，包括 C500 |
| 操作系统 | Linux 操作系统，包括 CentOS、Ubuntu、KylinV10 等 |

```

## 安装飞桨框架

### 安装方式一：wheel 包安装

沐曦曦云C500 支持插件式安装，需先安装飞桨 CPU 安装包，再安装飞桨 沐曦 插件包：

```bash
# 先安装飞桨 CPU 安装包
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu

# 再安装飞桨 曦云C500 插件包
pip install paddle-metax-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/sdaa
```
⚠️ 注意：nightly 版本为每日构建，可能存在不稳定性。如果需要更稳定的版本，建议使用 3.1 版本:https://www.paddlepaddle.org.cn/packages/stable/metax/paddle-metax-gpu/
### 安装方式二：源代码编译安装

在启动的 docker 容器中，先安装飞桨 CPU 安装包，再下载 PaddleCustomDevice 源码编译得到飞桨 C500 插件包。

```bash
# 下载 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 在 PaddleCUstomDevice 根目录下执行以下指令更新子模块代码
git submodule sync
git submodule update --init --recursive

# 进入硬件后端(沐曦 曦云C500)目录
cd backends/metax

# 先安装飞桨 CPU 安装包
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu

# 执行编译脚本
bash build.sh

# 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/*.whl --force-reinstall
```
⚠️ 注意：nightly 版本为每日构建，可能存在不稳定性。如果需要更稳定的版本，建议使用 3.1 版本。

```


## 如何卸载

请使用以下命令卸载 Paddle:

```bash
pip uninstall paddlepaddle paddle-metax-gpu
```
