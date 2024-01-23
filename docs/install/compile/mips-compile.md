# **龙芯下从源码编译**

## 环境准备

* **处理器：Loongson-3A R4 (Loongson-3A4000)**
* **操作系统：Loongnix release 1.0**
* **Python 版本 3.8/3.9/3.10 (64 bit)**
* **pip 或 pip3 版本 20.2.2+ (64 bit)**

本文以 Loongson-3A4000 为例，介绍 Paddle 在 MIPS 架构下的源码编译。

## 安装步骤

目前在 MIPS 龙芯处理器加龙芯国产化操作系统上安装 Paddle，只支持源码编译的方式，接下来详细介绍各个步骤。

<a name="mips_source"></a>
### **源码编译**

1. 龙芯操作系统`Loongnix release 1.0`默认安装的 gcc 版本是 4.9，但 yum 源提供了 gcc-7 的工具链，在此处安装 gcc-7。可以参考龙芯龙芯开源社区[文章](http://www.loongnix.org/index.php/Gcc7.3.0)

    ```
    sudo yum install devtoolset-7-gcc.mips64el devtoolset-7-gcc-c++.mips64el devtoolset-7.mips64el
    ```

    设置环境变量使得 gcc-7 生效

    ```
    source /opt/rh/devtoolset-7/enable
    ```

2. 龙芯系统自带的 python 都是基于 gcc4.9，在第 1 步时选择使用了 gcc-7.3，此处需要源码安装 Python，此处以 Python3.8 为例。

    ```
    sudo yum install libffi-devel.mips64el openssl-devel.mips64el libsqlite3x-devel.mips64el sqlite-devel.mips64el lbzip2-utils.mips64el lzma.mips64el tk.mips64el uuid.mips64el gdbm-devel.mips64el gdbm.mips64el openjpeg-devel.mips64el zlib-devel.mips64el libjpeg-turbo-devel.mips64el openjpeg-devel.mips64el
    ```

    ```
    wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz && tar xzf Python-3.8.0.tgz && cd Python-3.8.0
    ```

    ```
    ./configure –prefix $HOME/python38–enable−shared
    ```

    ```
    make -j
    ```

    ```
    make install
    ```

    设置环境变量，使得 python38 生效

    ```
    export PATH=$HOME/python38/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/python38/lib:$LD_LIBRARY_PATH
    ```

3. Paddle 依赖 cmake 进行编译构建，需要 cmake 版本>=3.15，龙芯操作系统源提供 cmake 的版本是 3.9，且尝试源码编译 cmake 失败，此处临时的处理方式是修改 Paddle 主目录的`CMakeLists.txt`, `cmake_minimum_required(VERSION 3.15)` 修改为 `cmake_minimum_required(VERSION 3.9)`。等到龙芯系统支持 cmake >= 3.15 后则不需要其它操作。


4. Paddle 内部使用 patchelf 来修改动态库的 rpath，操作系统提供的源包括了 patchelf，直接安装即可，后续会考虑在 MIPS 上移出该依赖。

    ```
    sudo yum install patchelf.mips64el
    ```

5. 将 Paddle 的源代码克隆到当下目录下的 Paddle 文件夹中，并进入 Paddle 目录

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    ```

    ```
    cd Paddle
    ```

6. 根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装 Python 依赖库。


7. 切换到`develop`分支下进行编译：

    ```
    git checkout develop
    ```

6. 并且请创建并进入一个叫 build 的目录下：

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
    cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MIPS=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_MKL=OFF
    ```

9. 编译。

    ```
    make -j$(nproc)
    ```

10. 编译成功后进入`Paddle/build/python/dist`目录下找到生成的`.whl`包。

11. 在当前机器或目标机器安装编译好的`.whl`包：

    ```
    python -m pip install -U（whl 包的名字）`或`python3 -m pip install -U（whl 包的名字）
    ```

恭喜，至此您已完成 PaddlePaddle 在龙芯环境下的编译安装。


## **验证安装**
安装完成后您可以使用 `python` 或 `python3` 进入 python 解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

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
python -m pip uninstall paddlepaddle
```
或
```
python3 -m pip uninstall paddlepaddle
```


## **备注**

已在 MIPS 架构下测试过 resnet50, mobilenetv1, ernie， ELMo 等模型，基本保证了预测使用算子的正确性，如果您在使用过程中遇到计算结果错误，编译失败等问题，请到[issue](https://github.com/PaddlePaddle/Paddle/issues)中留言，我们会及时解决。

预测文档见[doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html)，使用示例见[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)
