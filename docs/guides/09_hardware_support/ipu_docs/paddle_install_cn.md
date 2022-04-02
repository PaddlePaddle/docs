# 飞桨框架IPU版安装说明

飞桨框架IPU版支持基于IPU的Python的训练和原生推理，目前仅支持通过源代码编译安装。

## 通过源代码编译安装

建议在Docker环境内编译和使用飞桨框架IPU版，下面的说明将使用基于Ubuntu18.04的容器进行编译，使用的Python版本为Python3.7。

**第一步** 构建Docker镜像

```
# 下载源码
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 构建 Docker 镜像
docker build -t paddlepaddle/paddle:latest-dev-ipu \
-f tools/dockerfile/Dockerfile.ipu .
```

**第二步** 下载Paddle源码并编译

```
# 创建并运行 Docker 容器
# 需要将主机端的 ipuof 配置文件映射到容器中，可通过设置 HOST_IPUOF_PATH 环境变量传入
# 可以按照主机上的 ipuof 配置文件名称对下面的脚本进行修改，将 ipu.conf 改成相应的名称
# 更多关于 ipuof 配置的信息可访问 https://docs.graphcore.ai/projects/vipu-admin/en/latest/cli_reference.html?highlight=ipuof#ipuof-configuration-file
docker run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \
--device=/dev/infiniband/ --ipc=host \
--name paddle-dev-ipu -w /home \
-v ${HOST_IPUOF_PATH}:/ipuof \
-e IPUOF_CONFIG_PATH=/ipuof/ipu.conf \
-it paddlepaddle/paddle:latest-dev-ipu bash

# 容器内下载源码
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行 CMake
cmake .. -DWITH_IPU=ON -DWITH_PYTHON=ON -DPY_VERSION=3.7 -DWITH_MKL=ON \
         -DPOPLAR_DIR=/opt/poplar -DPOPART_DIR=/opt/popart -DCMAKE_BUILD_TYPE=Release

# 开始编译
make -j$(nproc)

# 安装编译生成的 wheel 包
pip install -U python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
```

**第三步** 验证安装

```
python -c "import paddle; paddle.utils.run_check()"
```
