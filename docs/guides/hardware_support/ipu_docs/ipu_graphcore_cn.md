# Graphcore IPU

IPU 是 Graphcore 推出的用于 AI 计算的专用芯片，PaddlePaddle IPU 版可以支持在 Graphcore IPU 上进行模型训练与推理。
## 系统要求

| 芯片类型 | 操作系统 | IPU SDK 版本 |
| ---- | ---- | ---- |
| Colossus MK2 GC200 IPU  | Ubuntu 18.04 | Poplar 2.5.1 |
## 安装说明

飞桨框架IPU版提供两种安装方式：

- Docker镜像方式启动
- 通过源代码编译安装

### Docker镜像方式启动

当前 Docker 镜像包含预编译的飞桨框架 IPU 版，镜像基于 Ubuntu18.04 基础镜像构建，内置的 Python 版本为 Python3.7。

**第一步**：拉取飞桨框架 IPU 版镜像

```bash
docker pull registry.baidubce.com/device/paddlepaddle:ipu-poplar251
```

**第二步**：构建并进入 Docker 容器

**注意**：容器启动命令需将主机端的 IPUoF 配置文件映射到容器中，可通过设置 IPUOF_CONFIG_PATH 环境变量指向 IPUoF 配置文件传入，更多关于 IPUoF 配置的信息请访问 [Graphcore: IPUoF configuration file](https://docs.graphcore.ai/projects/vipu-admin/en/latest/cli_reference.html?highlight=ipuof#ipuof-configuration-file)。

```bash
# 注意：这里的 --cap-add  --device 和 --ipc 等均需配置
export IPUOF_CONFIG_PATH=/opt/ipuof.conf
docker run -it --name paddle-ipu -v `pwd`:/workspace \
     --shm-size=128G --network=host --ulimit memlock=-1:-1 \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     --cap-add=IPC_LOCK --device=/dev/infiniband/ --ipc=host \
     -v ${IPUOF_CONFIG_PATH}:/ipuof.conf -e IPUOF_CONFIG_PATH=/ipuof.conf \
     registry.baidubce.com/device/paddlepaddle:ipu-poplar250 /bin/bash
```

**第三步**：检查容器运行环境

```bash
# 检查容器是否可以正确识别 IPU 设备
gc-info -l && gc-monitor
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
paddlepaddle-ipu       0.0.0.dev251

# 检查飞桨框架 IPU 版正常工作
python3 -c "import paddle; paddle.utils.run_check()"
# 预期得到如下结果
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle works well on 2 CPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

### 通过源代码编译安装

**预先要求**：请根据[编译依赖表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html)准备符合版本要求的依赖库，推荐使用飞桨官方镜像，或者根据 [Poplar SDK 文档](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html) 来准备相应的运行环境。

**第一步**：下载Paddle源码并编译，CMAKE编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

```bash
# 下载源码，默认 develop 分支
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行 CMake 命令进行配置
cmake .. -DPY_VERSION=3.7 -DWITH_IPU=ON -DWITH_MKL=ON \
         -DPOPLAR_DIR=/opt/poplar -DPOPART_DIR=/opt/popart \
         -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

# 执行 Make 命令进行编译
make -j$(nproc)
```

**第二步**：安装与验证编译生成的 whl 包

```bash
# 检查编译目录下生成的 Python whl 包
Paddle/build/python/dist/
└── paddlepaddle_ipu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 安装生成的 Python whl 包
python3 -m pip install -U paddlepaddle_ipu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 验证命令
python3 -c "import paddle; paddle.utils.run_check()"
# 预期得到如下结果
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle works well on 2 CPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 训练和推理示例

以下仅以 MINST 模型为例说明如果在 Graphcore IPU 上运行训练和推理。更多训练示例请参考[Paddle-BERT with Graphcore IPUs](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/bert/static_ipu), 更多推理示例请参考[Paddle Inference IPU Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++/ipu).

**第一步**：下载示例代码

```bash
# 下载示例代码，并解压
wget https://paddle-device.bj.bcebos.com/ipu/sample.tar.gz
tar -zxvf sample.tar.gz && cd sample

# 解压后的目录文件如下所示
sample/
├── image
│   └── infer_3.png
├── mnist_infer.py
├── mnist_train.py
├── model
└── README.md
```

**第二步**：运行训练示例

```bash
# 运行训练示例
python3 mnist_train.py
# 预期得到如下结果
start training
start compiling model for ipu, it will need some minutes
Graph compilation: 20/100
... ...
Graph compilation: 100/100
finish model compiling!
step: 0, loss: 2.798431158065796
step: 40, loss: 0.684889018535614
... ...
step: 920, loss: 0.3129602372646332
finish training!
start verifying
start compiling model for ipu, it will need some minutes
finish model compiling!
top1 score: 0.8819110576923077

# 训练完成后在 model 目录下会生成推理模型，目录结构如下
sample/
├── image
│   └── infer_3.png
├── mnist_infer.py
├── mnist_train.py
├── model
│   ├── mnist.pdiparams
│   └── mnist.pdmodel
└── README.md
```

**第三步**：运行推理示例

```bash
# 运行推理示例
python3 mnist_infer.py
# 预期得到如下结果
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
--- Running IR pass [inference_process_pass]
I0515 16:20:35.537405 25728 ir_analysis_pass.cc:46] argument has no fuse statis
--- Running analysis [ir_params_sync_among_devices_pass]
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0515 16:20:35.538944 25728 analysis_predictor.cc:1024] ======= optimize end =======
I0515 16:20:35.539006 25728 naive_executor.cc:102] ---  skip [feed], feed -> img
I0515 16:20:35.539033 25728 naive_executor.cc:102] ---  skip [save_infer_model/scale_0.tmp_0], fetch -> fetch
Inference result of ./infer_3.png is:  3
```

## 如何卸载

请使用以下命令卸载 PaddlePaddle IPU 版：

```bash
python3 -m pip uninstall paddlepaddle-ipu
```
