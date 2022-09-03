# 飞桨框架昇腾 NPU 版安装说明

飞桨框架 NPU 版支持基于华为鲲鹏 CPU 与昇腾 NPU 的 Python 的训练和原生推理。

### 环境准备

当前 Paddle 昇腾 910 NPU 版支持的华为 CANN 社区版 5.0.2.alpha005，请先根据华为昇腾 910 NPU 的要求，进行相关 NPU 运行环境的部署和配置，参考华为官方文档 [CANN 社区版安装指南](https://support.huaweicloud.com/instg-cli-cann502-alpha005/atlasdeploy_03_0002.html)。

Paddle 昇腾 910 NPU 版目前仅支持源码编译安装，其中编译与运行相关的环境要求如下：

- **CPU 处理器:** 鲲鹏 920
- **操作系统:** Ubuntu 18.04 / CentOS 7.6 / KylinV10SP1 / EulerOS 2.8
- **CANN 社区版:** 5.0.2.alpha005
- **Python 版本:** 3.7
- **CMake 版本:** 3.15+
- **GCC/G++版本:** 8.2+

## 安装方式：通过源码编译安装

**第一步**：准备 CANN 社区版 5.0.2.alpha005 运行环境 (推荐使用 Paddle 镜像)

可以直接从 Paddle 的官方镜像库拉取预先装有 CANN 社区版 5.0.2.alpha005 的 docker 镜像，或者根据 [CANN 社区版安装指南](https://support.huaweicloud.com/instg-cli-cann502-alpha005/atlasdeploy_03_0002.html) 来准备相应的开发与运行环境。

```bash
# 拉取镜像
docker pull paddlepaddle/paddle:latest-dev-cann5.0.2.alpha005-gcc82-aarch64

# 启动容器，注意这里的参数 --device，容器仅映射设备 ID 为 4 到 7 的 4 张 NPU 卡，如需映射其他卡相应增改设备 ID 号即可
docker run -it --name paddle-npu-dev -v /home/<user_name>:/workspace  \
            --pids-limit 409600 --network=host --shm-size=128G \
            --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
            --device=/dev/davinci4 --device=/dev/davinci5 \
            --device=/dev/davinci6 --device=/dev/davinci7 \
            --device=/dev/davinci_manager \
            --device=/dev/devmm_svm \
            --device=/dev/hisi_hdc \
            -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
            -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
            -v /usr/local/dcmi:/usr/local/dcmi \
            paddlepaddle/paddle:latest-dev-cann5.0.2.alpha005-gcc82-aarch64 /bin/bash

# 检查容器中是否可以正确识别映射的昇腾 DCU 设备
npu-smi info

# 预期得到类似如下的结果
+------------------------------------------------------------------------------------+
| npu-smi 1.9.3                    Version: 21.0.rc1                                 |
+----------------------+---------------+---------------------------------------------+
| NPU   Name           | Health        | Power(W)   Temp(C)                          |
| Chip                 | Bus-Id        | AICore(%)  Memory-Usage(MB)  HBM-Usage(MB)  |
+======================+===============+=============================================+
| 4     910A           | OK            | 67.2       30                               |
| 0                    | 0000:C2:00.0  | 0          303  / 15171      0    / 32768   |
+======================+===============+=============================================+
| 5     910A           | OK            | 63.8       25                               |
| 0                    | 0000:82:00.0  | 0          2123 / 15171      0    / 32768   |
+======================+===============+=============================================+
| 6     910A           | OK            | 67.1       27                               |
| 0                    | 0000:42:00.0  | 0          1061 / 15171      0    / 32768   |
+======================+===============+=============================================+
| 7     910A           | OK            | 65.5       30                               |
| 0                    | 0000:02:00.0  | 0          2563 / 15078      0    / 32768   |
+======================+===============+=============================================+
```

**第二步**：下载 Paddle 源码并编译，CMAKE 编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

```bash
# 下载源码
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行 cmake
cmake .. -DPY_VERSION=3.7 -DWITH_ASCEND=OFF -DWITH_ARM=ON -DWITH_ASCEND_CL=ON \
         -DWITH_ASCEND_INT64=ON -DWITH_DISTRIBUTE=ON -DWITH_TESTING=ON -DON_INFER=ON \
         -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 使用以下命令来编译
make TARGET=ARMV8 -j$(nproc)
```

**第三步**：安装与验证编译生成的 wheel 包

编译完成之后进入`Paddle/build/python/dist`目录即可找到编译生成的.whl 安装包，安装与验证命令如下：

```bash
# 安装命令
python -m pip install -U paddlepaddle_npu-0.0.0-cp37-cp37m-linux_aarch64.whl

# 验证命令
python -c "import paddle; paddle.utils.run_check()"

# 预期得到类似以下结果：
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 NPU.
PaddlePaddle works well on 4 NPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 如何卸载

请使用以下命令卸载 Paddle:

```bash
pip uninstall paddlepaddle-npu
```
