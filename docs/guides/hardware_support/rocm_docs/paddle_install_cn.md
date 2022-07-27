# 飞桨框架 ROCm 版安装说明

飞桨框架 ROCm 版支持基于海光 CPU 和 DCU 的 Python 的训练和原生预测，当前支持的 ROCm 版本为 4.0.1, 提供两种安装方式：

- 通过预编译的 wheel 包安装
- 通过源代码编译安装

## 安装方式一：通过 wheel 包安装

**注意**：当前仅提供基于 CentOS 7.8 & ROCm 4.0.1 的 docker 镜像，与 Python 3.7 的 wheel 安装包。

**第一步**：准备 ROCm 4.0.1 运行环境 (推荐使用 Paddle 镜像)

可以直接从 Paddle 的官方镜像库拉取预先装有 ROCm 4.0.1 的 docker 镜像，或者根据 [ROCm 安装文档](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#centos-rhel) 来准备相应的运行环境。

```bash
# 拉取镜像
docker pull paddlepaddle/paddle:latest-dev-rocm4.0-miopen2.11

# 启动容器，注意这里的参数，例如 shm-size, device 等都需要配置
docker run -it --name paddle-rocm-dev --shm-size=128G \
     --device=/dev/kfd --device=/dev/dri --group-add video \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     paddlepaddle/paddle:latest-dev-rocm4.0-miopen2.11 /bin/bash

# 检查容器是否可以正确识别海光 DCU 设备
rocm-smi

# 预期得到以下结果：
======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK     MCLK    Fan   Perf  PwrCap  VRAM%  GPU%
0    50.0c  23.0W   1319Mhz  800Mhz  0.0%  auto  300.0W    0%   0%
1    48.0c  25.0W   1319Mhz  800Mhz  0.0%  auto  300.0W    0%   0%
2    48.0c  24.0W   1319Mhz  800Mhz  0.0%  auto  300.0W    0%   0%
3    49.0c  27.0W   1319Mhz  800Mhz  0.0%  auto  300.0W    0%   0%
================================================================================
============================= End of ROCm SMI Log ==============================
```

**第二步**：下载 Python3.7 wheel 安装包

```bash
pip install --pre paddlepaddle-rocm -f https://www.paddlepaddle.org.cn/whl/rocm/develop.html
```

**第三步**：验证安装包

安装完成之后，运行如下命令。如果出现 PaddlePaddle is installed successfully!，说明已经安装成功。

```bash
python -c "import paddle; paddle.utils.run_check()"
```

## 安装方式二：通过源码编译安装

**注意**：当前 Paddle 只支持 CentOS 7.8 & ROCm 4.0.1 编译环境，且根据 ROCm 4.0.1 的需求，支持的编译器为 devtoolset-7。

**第一步**：准备 ROCm 4.0.1 编译环境 (推荐使用 Paddle 镜像)

可以直接从 Paddle 的官方镜像库拉取预先装有 ROCm 4.0.1 的 docker 镜像，或者根据 [ROCm 安装文档](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#centos-rhel) 来准备相应的运行环境。

```bash
# 拉取镜像
docker pull paddlepaddle/paddle:latest-dev-rocm4.0-miopen2.11

# 启动容器，注意这里的参数，例如 shm-size, device 等都需要配置
docker run -it --name paddle-rocm-dev --shm-size=128G \
     --device=/dev/kfd --device=/dev/dri --group-add video \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     paddlepaddle/paddle:latest-dev-rocm4.0-miopen2.11 /bin/bash

# 检查容器是否可以正确识别海光 DCU 设备
rocm-smi

# 预期得到以下结果：
======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK     MCLK    Fan   Perf  PwrCap  VRAM%  GPU%
0    50.0c  23.0W   1319Mhz  800Mhz  0.0%  auto  300.0W    0%   0%
1    48.0c  25.0W   1319Mhz  800Mhz  0.0%  auto  300.0W    0%   0%
2    48.0c  24.0W   1319Mhz  800Mhz  0.0%  auto  300.0W    0%   0%
3    49.0c  27.0W   1319Mhz  800Mhz  0.0%  auto  300.0W    0%   0%
================================================================================
============================= End of ROCm SMI Log ==============================
```

请在编译之前，检查如下的环境变量是否正确，如果没有则需要安装相应的依赖库，并导出相应的环境变量。以 Paddle 官方的镜像举例，环境变量如下：

```bash
# PATH 与 LD_LIBRARY_PATH 中存在 devtoolset-7，如果没有运行以下命令
source /opt/rh/devtoolset-7/enable

# PATH 中存在 cmake 3.16.0
export PATH=/opt/cmake-3.16/bin:${PATH}

# PATH 与 LD_LIBRARY_PATH 中存在 rocm 4.0.1
export PATH=/opt/rocm/opencl/bin:/opt/rocm/bin:${PATH}
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}

# PATH 中存在 Python 3.7
# 注意：镜像中的 python 3.7 通过 miniconda 安装，请通过 conda activate base 命令加载 Python 3.7 环境
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
cmake .. -DPY_VERSION=3.7 -DWITH_ROCM=ON -DWITH_TESTING=ON -DWITH_DISTRIBUTE=ON \
         -DWITH_MKL=ON -DCMAKE_BUILD_TYPE=Release

# 使用以下命令来编译
make -j$(nproc)
```

**第三步**：安装与验证编译生成的 wheel 包

编译完成之后进入`Paddle/build/python/dist`目录即可找到编译生成的.whl 安装包，安装与验证命令如下：

```bash
# 安装命令
python -m pip install -U paddlepaddle_rocm-0.0.0-cp37-cp37m-linux_x86_64.whl

# 验证命令
python -c "import paddle; paddle.utils.run_check()"
```

## 如何卸载

请使用以下命令卸载 Paddle:

```bash
pip uninstall paddlepaddle-rocm
```
