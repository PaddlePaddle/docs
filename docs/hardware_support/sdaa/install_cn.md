# 太初 SDAA 安装说明

飞桨框架 SDAA 版支持太初 SDAA 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译安装得到 wheel 包

## 太初 SDAA 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 太初元碁 系列芯片，包括 T100 |
| 操作系统 | Linux 操作系统，包括 CentOS、Ubuntu、KylinV10 等 |

## 运行环境准备

推荐使用太初官方发布的太初 SDAA 开发镜像，该镜像预装有太初 SDAA 基础运行环境库。

```bash
# 拉取镜像
wget http://mirrors.tecorigin.com/repository/teco-docker-tar-repo/release/ubuntu22.04/x86_64/2.1.0/paddle-2.1.0-paddle_sdaa2.1.0.tar
docker load < paddle-2.1.0-paddle_sdaa2.1.0.tar
```

```bash
# 启动容器
docker run -itd --name="paddle-SDAA-dev" -v $(pwd):/work --net=host \
 --device=/dev/tcaicard0 --device=/dev/tcaicard1 --device=/dev/tcaicard2 --device=/dev/tcaicard3 \
 --cap-add SYS_PTRACE --cap-add SYS_ADMIN --shm-size 128g \
 jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/paddle:2.1.0-paddle_sdaa2.1.0 /bin/bash
```

#### 选项说明及可调整参数

##### ① `--name paddle-SDAA-dev`
- **作用**：指定容器名称。
- **可调整**：
  - 用户可改为其他名称，例如 `paddle-SDAA-test`，方便区分不同实验。

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
# 检查容器内是否正常识别太初 SDAA 设备
teco-smi
```

```bash
# 预期输出
+-----------------------------------------------------------------------------+
|  TECO-SMI: 1.12.0        SDAADriver: 2.1.0        SDAARuntime: 2.1.0        |
|-------------------------------+----------------------+----------------------|
| Index  Name                   | Bus-Id               | Health      SPE-Util |
|        Temp          Pwr Usage|          Memory-Usage|                      |
|=============================================================================|
|   0    TECO_AICARD_01         | 00000000:4F:00.0     | OK                0% |
|        35C               190W |        0MB / 16384MB |                      |
|-------------------------------+----------------------+----------------------|
|   1    TECO_AICARD_01         | 00000000:4F:00.0     | OK                0% |
|        35C               190W |        0MB / 16384MB |                      |
|-------------------------------+----------------------+----------------------|
|   2    TECO_AICARD_01         | 00000000:4F:00.0     | OK                0% |
|        35C               190W |        0MB / 16384MB |                      |
|-------------------------------+----------------------+----------------------|
|   3    TECO_AICARD_01         | 00000000:52:00.0     | OK                0% |
|        37C                94W |        0MB / 16384MB |                      |
+-------------------------------+----------------------+----------------------+
```

## 安装飞桨框架

### 安装方式一：wheel 包安装

SDAA 支持插件式安装，需先安装飞桨 CPU 安装包，再安装飞桨 SDAA 插件包。在启动的 docker 容器中，执行以下命令：

```bash
# 先安装飞桨 CPU 安装包
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu

# 再安装飞桨 SDAA 插件包
pip install paddle-sdaa -i https://www.paddlepaddle.org.cn/packages/nightly/sdaa
```
⚠️ 注意：nightly 版本为每日构建，可能存在不稳定性。如果需要更稳定的版本，建议使用 3.0 版本:https://www.paddlepaddle.org.cn/packages/stable/sdaa/paddle-sdaa/
### 安装方式二：源代码编译安装

在启动的 docker 容器中，先安装飞桨 CPU 安装包，再下载 PaddleCustomDevice 源码编译得到飞桨 SDAA 插件包。

```bash
# 下载 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 在 PaddleCUstomDevice 根目录下执行以下指令更新子模块代码
git submodule sync
git submodule update --init --recursive

# 进入硬件后端(太初 SDAA)目录
cd backends/sdaa

# 先安装飞桨 CPU 安装包
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu

# 执行编译脚本
bash compile.sh

# 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/*.whl --force-reinstall
```
⚠️ 注意：nightly 版本为每日构建，可能存在不稳定性。如果需要更稳定的版本，建议使用 3.0 版本。
## 基础功能检查

安装完成后，在 docker 容器中输入如下命令进行飞桨基础健康功能的检查。

```bash
# 列出可用硬件后端
python3 -c "import paddle; print(paddle.device.get_all_custom_device_type())"
```
```bash
# 预期得到如下输出结果
['sdaa']
```
```bash
# 使用 paddle_sdaa utils 模块的 `run_check` 功能检查 paddle-sdaa 插件和 PaddlePaddle 主框架是否正常安装
python3 -c "import paddle_sdaa; paddle_sdaa.utils.run_check()"
```
```bash
# 预期得到输出如下
+--------------+---------------------+-----------------+
|  Dependence  | Compilation Version | Current Version |
+--------------+---------------------+-----------------+
| sdaa_driver  |        2.1.0        |      2.1.0      |
| sdaa_runtime |        2.1.0        |      2.1.0      |
+--------------+---------------------+-----------------+
```
## 如何卸载

请使用以下命令卸载 Paddle:

```bash
pip uninstall paddlepaddle paddle-sdaa
```
