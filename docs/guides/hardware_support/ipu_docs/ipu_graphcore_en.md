# Graphcore IPU Support

IPU(Intelligence Processing Unit) is a microprocessor specialised for processing machine learning workloads, pioneered by Graphcore. PaddlePaddle supports training and inference on IPU.

## Supported Platforms

| IPU Chip Type | Operating System | IPU SDK Version |
| ---- | ---- | ---- |
| Colossus MK2 GC200 IPU  | Ubuntu 18.04 | Poplar 2.5.1 |

## Installation

Two installation methods supported as follows:

- Installation via Docker
- Building from Source Code

### Installation via Docker

PaddlePaddle published the docker image with pre-build paddlepaddle-ipu wheel package installed.

**Step 1**. Pull PaddlePaddle IPU docker image

```bash
# Pull PaddlePaddle IPU docker image
docker pull registry.baidubce.com/device/paddlepaddle:ipu-poplar251
```

**Step 2**. Start a new container based on the new image

**Note**: The docker run command requires the IPUoF configuration file to detect IPU devices in container. Please set the environment variable IPUOF_CONFIG_PATH with the path of IPUoF configuration file. For more information about IPUoF configuration file, please refer to [Graphcore: IPUoF configuration file](https://docs.graphcore.ai/projects/vipu-admin/en/latest/cli_reference.html?highlight=ipuof#ipuof-configuration-file).

```bash
# Note: --cap-add  --device and --ipc options are requried
export IPUOF_CONFIG_PATH=/opt/ipuof.conf
docker run -it --name paddle-ipu -v `pwd`:/workspace \
     --shm-size=128G --network=host --ulimit memlock=-1:-1 \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     --cap-add=IPC_LOCK --device=/dev/infiniband/ --ipc=host \
     -v ${IPUOF_CONFIG_PATH}:/ipuof.conf -e IPUOF_CONFIG_PATH=/ipuof.conf \
     registry.baidubce.com/device/paddlepaddle:ipu-poplar251 /bin/bash
```

**Step 3**. Verify paddlepaddle-ipu program

```bash
# Verify IPU can be monitored inside container
gc-info -l && gc-monitor
# Expected to get output as following
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

# Verify paddlepaddle-ipu is installed
pip list | grep paddlepaddle-ipu
# Expected to get output as following
paddlepaddle-ipu       0.0.0.dev251

# Verify paddlepaddle-ipu works successfully
python3 -c "import paddle; paddle.utils.run_check()"
# Expected to get output as follows
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle works well on 2 CPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

### Building from Source Code

**Prerequisite**：Please prepare the compiling environment based on [Compile Dependency Table](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/Tables_en.html). Propose to use docker container environment in previous section, or prepare the compiling enviroment based on [Poplar SDK Document](https://docs.graphcore.ai/projects/ipu-pod-getting-started/en/latest/installation.html).

**Step 1**. Download PaddlePaddle source code and compile, refer to [Compile Option Table](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/Tables_en.html) for more CMake options

```bash
# Download PaddlePaddle source code, default branch is develop
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# Prepare build directory
mkdir build && cd build

# CMake command to configure
cmake .. -DPY_VERSION=3.7 -DWITH_IPU=ON -DWITH_MKL=ON \
         -DPOPLAR_DIR=/opt/poplar -DPOPART_DIR=/opt/popart \
         -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

# Make command to compile
make -j$(nproc)
```

**Step 2**. Install and verify the built whl package

```bash
# Check built whl package is generated
Paddle/build/python/dist/
└── paddlepaddle_ipu-0.0.0-cp37-cp37m-linux_x86_64.whl

# Install the built whl package
python3 -m pip install -U paddlepaddle_ipu-0.0.0-cp37-cp37m-linux_x86_64.whl

# Verification of paddlepaddle-ipu program
python3 -c "import paddle; paddle.utils.run_check()"
# Expected to get output as follows
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle works well on 2 CPUs.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## Training and Inference Demo

This section is a quick start demo of train and inference MNIST model on IPU. For more training example please refer to [Paddle-BERT with Graphcore IPUs](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/bert/static_ipu), and for more inference example please refer to [Paddle Inference IPU Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c++/ipu).

**Step 1**. Download sample code

```bash
# Download and unzip sample code for paddlepaddle-ipu
wget https://paddle-device.bj.bcebos.com/ipu/sample.tar.gz
tar -zxvf sample.tar.gz && cd sample

# list contents of sample as following after unzip
sample/
├── image
│   └── infer_3.png
├── mnist_infer.py
├── mnist_train.py
├── model
└── README.md
```

**Step 2**. Run training demo on IPU

```bash
# Run minst train
python3 mnist_train.py
# Expected to get output as follows
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

# Generated inference model will be saved under ./model directory
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

**Step 3**. Run inference demo on IPU

```bash
# Run minst infer
python3 mnist_infer.py
# Expected to get output as follows
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

## Uninstallation of PaddlePaddle on IPU

Please use the following command to uninstall PaddlePaddle on IPU:

```bash
python3 -m pip uninstall paddlepaddle-ipu
```
