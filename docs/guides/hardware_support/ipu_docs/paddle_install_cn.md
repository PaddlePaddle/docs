# 飞桨框架 IPU 版安装说明

飞桨框架 IPU 版支持基于 Graphcore IPU 的 Python 的训练和原生推理，当前支持的 Poplar 版本为 2.5.0, 提供两种安装方式：

- Docker 镜像方式启动
- 通过源代码编译安装

## Docker 镜像方式启动

当前 Docker 镜像包含预编译的飞桨框架 IPU 版，镜像基于 Ubuntu18.04 基础镜像构建，内置的 Python 版本为 Python3.8。

**第一步**：拉取飞桨框架 IPU 版镜像

```bash
docker pull registry.baidubce.com/device/paddlepaddle:ipu-poplar250
```

**第二步**：构建并进入 Docker 容器

**注意**：容器启动命令需将主机端的 IPUoF 配置文件映射到容器中，可通过设置 IPUOF_CONFIG_PATH 环境变量指向 IPUoF 配置文件传入，更多关于 IPUoF 配置的信息请访问 [Graphcore: IPUoF configuration file](https://docs.graphcore.ai/projects/vipu-admin/en/latest/cli_reference.html?highlight=ipuof#ipuof-configuration-file)。

```bash
# 注意替换这里的 /home/<username> 到对应的用户目录
export IPUOF_CONFIG_PATH=/opt/ipuof.conf
docker run -it --name paddle-ipu -v /home/<username>:/workspace \
     --shm-size=128G --network=host --ulimit memlock=-1:-1 \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     --cap-add=IPC_LOCK --device=/dev/infiniband/ --ipc=host \
     -v ${IPUOF_CONFIG_PATH}:/ipuof.conf -e IPUOF_CONFIG_PATH=/ipuof.conf \
     registry.baidubce.com/device/paddlepaddle:ipu-poplar250 /bin/bash
```

**第三步**：检查容器运行环境

```bash
# 检查容器是否可以正确识别 IPU 设备
gc-monitor
# 预期得到如下结果
+---------------+--------------------------------------------------------------------------------+
|  gc-monitor   |              Partition: ipuof [active] has 4 reconfigurable IPUs               |
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
|    IPU-M    |       Serial       |IPU-M SW|Server version|  ICU FW  | Type | ID | IPU# |Routing|
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
|...31.100.130| 0134.0002.8210321  |        |    1.8.1     |  2.3.5   |M2000 | 0  |  3   |  DNC  |
|...31.100.130| 0134.0002.8210321  |        |    1.8.1     |  2.3.5   |M2000 | 1  |  2   |  DNC  |
|...31.100.130| 0134.0001.8210321  |        |    1.8.1     |  2.3.5   |M2000 | 2  |  1   |  DNC  |
|...31.100.130| 0134.0001.8210321  |        |    1.8.1     |  2.3.5   |M2000 | 3  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
+--------------------------------------------------------------------------------------------------+
|                             No attached processes in partition ipuof                             |
+--------------------------------------------------------------------------------------------------+

# 检查飞桨框架 IPU 版已经安装
pip list | grep paddlepaddle-ipu
# 预期得到如下结果
paddlepaddle-ipu       0.0.0.dev250

# 检查飞桨框架 IPU 版正常工作
python -c "import paddle; paddle.utils.run_check()"
# 预期得到如下结果
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle works well on 2 CPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 通过源代码编译安装

**预先要求**：建议在 Docker 环境内进行飞桨框架 IPU 版的源码编译，容器环境配置和启动命令请参考上一章节的内容。

**第一步**：检查容器编译环境

请在编译之前，检查如下的环境变量是否正确，如果没有则需要安装相应的依赖库，并导出相应的环境变量。

```bash
# PATH 中存在 GCC/G++ 8.2
export PATH=/opt/compiler/gcc-8.2/bin:${PATH}

# PATH 中存在 cmake 3.18.0
export PATH=/opt/cmake-3.18/bin:${PATH}

# PATH 与 LD_LIBRARY_PATH 中存在 popart 与 poplar
export PATH=/opt/popart/bin:/opt/poplar/lib:${PATH}
export LD_LIBRARY_PATH=/opt/popart/lib:/opt/poplar/lib:${LD_LIBRARY_PATH}

# PATH 中存在 Python 3.8
# 注意：镜像中的 python 3.8 通过 miniconda 安装，请通过 conda activate base 命令加载 Python 3.8 环境
export PATH=/opt/conda/bin:${PATH}
```

**第二步**：下载 Paddle 源码并编译，CMAKE 编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

```bash
# 下载源码，默认 develop 分支
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行 cmake
cmake .. -DPY_VERSION=3.8 -DWITH_IPU=ON -DWITH_MKL=ON \
         -DPOPLAR_DIR=/opt/poplar -DPOPART_DIR=/opt/popart \
         -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release

# 使用以下命令来编译
make -j$(nproc)
```

**第三步**：安装与验证编译生成的 whl 包

编译完成之后进入 `Paddle/build/python/dist` 目录即可找到编译生成的 .whl 安装包，安装与验证命令如下：

```bash
# 安装命令
python -m pip install -U paddlepaddle_ipu-0.0.0-cp38-cp38m-linux_x86_64.whl

# 验证命令
python -c "import paddle; paddle.utils.run_check()"
# 预期得到如下结果
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle works well on 2 CPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
## 如何卸载

请使用以下命令卸载 Paddle：

```bash
pip uninstall paddlepaddle-ipu
```
