# 飞桨框架IPU版安装说明

## 下载源码

```
git clone -b paddle_bert_release https://github.com/graphcore/Paddle.git
```

## 构建 docker 镜像

```
docker build -t paddlepaddle/paddle:latest-dev-ipu \

-f tools/dockerfile/Dockerfile.ipu .
```

## 创建并运行 docker container

## The ipuof.conf is required here.

```
docker run --ulimit memlock=-1:-1 --net=host --cap-add=IPC_LOCK \

--device=/dev/infiniband/ --ipc=host --name paddle-ipu-dev \

-v ${HOST_IPUOF_PATH}:/ipuof \

-e IPUOF_CONFIG_PATH=/ipuof/ipu.conf \

-it paddlepaddle/paddle:latest-dev-ipu bash
```

## 编译 PaddlePaddle

```
git clone -b develop https://github.com/PaddlePaddle/Paddle.git
```

```
cd Paddle
```

```
cmake -DPYTHON_EXECUTABLE=/usr/bin/python \

-DWITH_PYTHON=ON -DWITH_IPU=ON -DPOPLAR_DIR=/opt/poplar \

-DPOPART_DIR=/opt/popart -G "Unix Makefiles" -H`pwd` -B`pwd`/build
```

```
cmake --build \
`pwd`/build --config Release --target paddle_python -j$(nproc)
```


#安装与验证编译生成的wheel包

## 安装命令

```
pip install -U build/python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
```

## 验证安装

```
python -c "import paddle; print(paddle.fluid.is_compiled_with_ipu())"
```

## 预期得到以下结果：

```
> True
```
