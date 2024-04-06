# **兆芯下从源码编译**

## 环境准备

* **处理器：ZHAOXIN KaiSheng KH-37800D**
* **操作系统：centos7**
* **Python 版本 3.8/3.9/3.10 (64 bit)**
* **pip 或 pip3 版本 9.0.1+ (64 bit)**

兆芯为 x86 架构，编译方法与[Linux 下从源码编译 cpu 版](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/compile_Linux.html)一致。

## 安装步骤

本文在 ZHAOXIN 处理器下安装 Paddle，接下来详细介绍各个步骤。

<a name="zhaoxin_source"></a>
### **源码编译**

1. Paddle 依赖 cmake 进行编译构建，需要 cmake 版本>=3.15，如果操作系统提供的源包括了合适版本的 cmake，直接安装即可，否则需要[源码安装](https://github.com/Kitware/CMake)

    ```
    wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8.tar.gz
    ```

    ```
    tar -xzf cmake-3.16.8.tar.gz && cd cmake-3.16.8
    ```

    ```
    ./bootstrap && make && sudo make install
    ```

2. Paddle 内部使用 patchelf 来修改动态库的 rpath，如果操作系统提供的源包括了 patchelf，直接安装即可，否则需要源码安装，请参考[patchelf 官方文档](https://github.com/NixOS/patchelf)。

    ```
    ./bootstrap.sh
    ```

    ```
    ./configure
    ```

    ```
    make
    ```

    ```
    make check
    ```

    ```
    sudo make install
    ```

3. 将 Paddle 的源代码克隆到当下目录下的 Paddle 文件夹中，并进入 Paddle 目录

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    ```

    ```
    cd Paddle
    ```

4. 切换到`develop`分支下进行编译：

    ```
    git checkout develop
    ```

5. 根据[requirements.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装 Python 依赖库。

    ```
    pip install -r python/requirements.txt
    ```

6. 请创建并进入一个叫 build 的目录下：

    ```
    mkdir build && cd build
    ```

7. 链接过程中打开文件数较多，可能超过系统默认限制导致编译出错，设置进程允许打开的最大文件数：

    ```
    ulimit -n 4096
    ```

8. 执行 cmake：

    >具体编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

    For Python3:
    ```
    cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MKL=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_PYTHON=ON
    ```

9. 编译。

    ```
    make -j$(nproc)
    ```

10. 编译成功后进入`Paddle/build/python/dist`目录下找到生成的`.whl`包。

11. 在当前机器或目标机器安装编译好的`.whl`包：

    ```
    python3 -m pip install -U（whl 包的名字）
    ```

恭喜，至此您已完成 PaddlePaddle 在 FT 环境下的编译安装。


## **验证安装**
安装完成后您可以使用 `python` 或 `python3` 进入 python 解释器，输入
```
import paddle
```
再输入
```
paddle.utils.run_check()
```

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

在 mobilenetv1 和 resnet50 模型上测试

```
wget -O profile.tar https://paddle-cetc15.bj.bcebos.com/profile.tar?authorization=bce-auth-v1/4409a3f3dd76482ab77af112631f01e4/2020-10-09T10:11:53Z/-1/host/786789f3445f498c6a1fd4d9cd3897ac7233700df0c6ae2fd78079eba89bf3fb
```
```
tar xf profile.tar && cd profile
```
```
python resnet.py --model_file ResNet50_inference/model --params_file ResNet50_inference/params
# 正确输出应为：[0.0002414  0.00022418 0.00053661 0.00028639 0.00072682 0.000213
#              0.00638718 0.00128127 0.00013535 0.0007676 ]
```
```
python mobilenetv1.py --model_file mobilenetv1/model --params_file mobilenetv1/params
# 正确输出应为：[0.00123949 0.00100392 0.00109539 0.00112206 0.00101901 0.00088412
#              0.00121536 0.00107679 0.00106071 0.00099605]
```
```
python ernie.py --model_dir ernieL3H128_model/
# 正确输出应为：[0.49879393 0.5012061 ]
```

## **如何卸载**
请使用以下命令卸载 PaddlePaddle：

```
python3 -m pip uninstall paddlepaddle
```
或
```
python3 -m pip uninstall paddlepaddle
```

## **备注**

已在 ZHAOXIN 下测试过 resnet50, mobilenetv1, ernie， ELMo 等模型，基本保证了预测使用算子的正确性，如果您在使用过程中遇到计算结果错误，编译失败等问题，请到[issue](https://github.com/PaddlePaddle/Paddle/issues)中留言，我们会及时解决。

预测文档见[doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html)，使用示例见[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)
