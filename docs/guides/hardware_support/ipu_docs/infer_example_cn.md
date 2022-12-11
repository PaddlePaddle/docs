# 飞桨框架 IPU 版预测示例

飞桨框架 IPU 版支持飞桨原生推理库(Paddle Inference)，适用于云端推理。

## C++预测示例

**第一步**：源码编译 C++预测库

当前 Paddle IPU 版只支持通过源码编译的方式提供 C++预测库，编译环境准备请参考 [飞桨框架 IPU 版安装说明](./paddle_install_cn.html)。

```
# 下载源码
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行 CMake，注意这里需打开预测优化选项 ON_INFER
cmake .. -DWITH_IPU=ON -DWITH_PYTHON=ON -DPY_VERSION=3.7 -DWITH_MKL=ON -DON_INFER=ON \
         -DPOPLAR_DIR=/opt/poplar -DPOPART_DIR=/opt/popart -DCMAKE_BUILD_TYPE=Release

# 开始编译
make -j$(nproc)
```

成功编译后，C++预测库将存放于 `build/paddle_inference_install_dir` 目录下。

**第二步**：获取预测示例代码并编译运行

```
# 获取示例代码
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo
```

将获得的 C++预测库拷贝并重命名一份到 `Paddle-Inference-Demo/c++/lib/paddle_inference`。

```
cd Paddle-Inference-Demo/c++/paddle-ipu

# 编译
bash ./compile.sh

# 运行
bash ./run.sh
```
