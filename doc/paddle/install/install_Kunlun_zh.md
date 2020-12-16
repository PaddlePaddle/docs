# 支持昆仑XPU的Paddle安装

## 安装方式一：通过Wheel包安装

安装预编译的支持昆仑XPU的wheel包，目前此wheel包只支持Ubuntu环境，其他环境请选择安装方式二，源码编译安装。

#### 下载安装包

Python3.7

o  ```wget https://paddle-wheel.bj.bcebos.com/kunlun/paddlepaddle-2.0.0rc1-cp37-cp37m-linux_x86_64.whl```

o  ```python3.7 -m pip install -U paddlepaddle-2.0.0rc1-cp37-cp37m-linux_x86_64.whl ```

Python2.7

o  ```wget https://paddle-wheel.bj.bcebos.com/kunlun/paddlepaddle-2.0.0rc1-cp27-cp27mu-linux_x86_64.whl```

o  ```python2.7 -m pip install -U paddlepaddle-2.0.0rc1-cp27-cp27mu-linux_x86_64.whl```



#### 验证安装

安装完成后您可以使用 python 或 python3 进入python解释器，输入

```import paddle ```

再输入

``` paddle.utils.run_check()```

如果出现PaddlePaddle is installed successfully!，说明您已成功安装。



#### 训练示例

下载并运行示例：

o  ```wget https://fleet.bj.bcebos.com/kunlun/mnist_example.py ```

o  ```python mnist_example.py --use_device=xpu --num_epochs=5```

注：后端有cpu/gpu/xpu三种设备可以自行配置


#### 如何卸载

请使用以下命令卸载PaddlePaddle：

 ```pip uninstall paddlepaddle```

或

 ```pip3 uninstall paddlepaddle ```

另外，如果使用预编译的支持昆仑XPU的wheel包出现环境问题，推荐使用源码自行编译支持昆仑XPU的包。





## 安装方式二：从源码编译支持昆仑XPU的包

#### 环境准备

- **处理器：****Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz**
- **操作系统：****Ubuntu 16.04.6 LTS**
- **Python** **版本**     **2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**
- **pip** **或** **pip3** **版本** **9.0.1+ (64 bit)**
- **cmake**  **版本** **3.10+**
- **gcc/g++**  **版本** **8.2+**

#### 安装步骤

目前在昆仑 XPU机器上面运行Paddle，必须要先安装从源码编译支持昆仑XPU的Paddle包，接下来详细介绍各个步骤。

##### **源码编译**

1. Paddle依赖cmake进行编译构建，需要cmake版本>=3.10，如果操作系统提供的源包括了合适版本的cmake，直接安装即可，否则需要

   o  ```wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8.tar.gz```

   o  ```tar -xzf cmake-3.16.8.tar.gz && cd cmake-3.16.8 ```

   o  ```./bootstrap && make && sudo make install```

2. Paddle内部使用patchelf来修改动态库的rpath，如果操作系统提供的源包括了patchelf，直接安装即可，否则需要源码安装，请参考

   o  ```./bootstrap.sh ```

   o ``` ./configure ```

   o ``` make ```

   o ``` make check ```

   o  ```sudo make install```

3. 根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装Python依赖库。

4. 将Paddle的源代码克隆到当下目录下的Paddle文件夹中，并进入Paddle目录

   o ``` git clone https://github.com/PaddlePaddle/Paddle.git```

   o  ```cd Paddle```

5. 切换到较稳定release分支下进行编译：

   ```git checkout [分支名]```

   例如：

   ```git checkout release/2.0-rc1```

   目前仅有2.0-rc1及其以后发版的分支支持昆仑XPU。

6. 并且请创建并进入一个叫build的目录下：

   ```mkdir build && cd build```

7. 链接过程中打开文件数较多，可能超过系统默认限制导致编译出错，设置进程允许打开的最大文件数：

   ```ulimit -n 4096```

8. 执行cmake：
9. 具体编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/install/quick/Tables.html#Compile)

For Python2: ```cmake .. -DPY_VERSION=2 -DPYTHON_EXECUTABLE=`which python2` -DWITH_MKL=OFF -DWITH_XPU=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release ```

For Python3: ```cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MKL=OFF -DWITH_XPU=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release ```

10. 使用以下命令来编译：

    ```make -j$(nproc)```

11. 编译成功后进入Paddle/build/python/dist目录下找到生成的.whl包 。

12. 将生成的.whl包copy至带有昆仑XPU的目标机器上，并在目标机器上根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装Python依赖库。（如果编译机器同时为带有昆仑XPU的目标机器，略过此步）

13. 在带有昆仑XPU的目标机器安装编译好的.whl包：pip install -U（whl包的名字）或pip3 install -U（whl包的名字）恭喜，至此您已完成昆仑XPU机器上PaddlePaddle的编译安装。



#### 验证安装

安装完成后您可以使用 python 或 python3 进入python解释器，输入

```import paddle ```

再输入

``` paddle.utils.run_check()```

如果出现PaddlePaddle is installed successfully!，说明您已成功安装。



#### 训练示例

下载并运行示例：

o  ```wget https://fleet.bj.bcebos.com/kunlun/mnist_example.py ```

o  ```python mnist_example.py --use_device=xpu --num_epochs=5```

注：后端有cpu/gpu/xpu三种设备可以自行配置

#### 如何卸载

请使用以下命令卸载PaddlePaddle：

 ```pip uninstall paddlepaddle```

或

 ```pip3 uninstall paddlepaddle ```

另外，如果使用预编译的支持昆仑XPU的wheel包出现环境问题，推荐使用源码自行编译支持昆仑XPU的包。


## 附录：当前昆仑XPU适配支持的模型

|  模型名称  | 模型地址  |
|  ----  | ----  |
| ResNet50  | [模型地址](https://github.com/PaddlePaddle/PaddleClas/tree/dygraph/docs/zh_CN/extension/train_on_xpu.md) |
| MobileNetv3  | [模型地址](https://github.com/PaddlePaddle/PaddleClas/tree/dygraph/docs/zh_CN/extension/train_on_xpu.md) |
| Deeplabv3  | [模型地址](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/legacy/docs/usage.md) |
| DQN  | [模型地址](https://github.com/PaddlePaddle/PARL/blob/develop/examples/DQN/README.md) |
| Bertbase  | [模型地址](https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/legacy/pretrain_language_models/BERT/README.md) |
