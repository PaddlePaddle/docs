# **飞腾下从源码编译**

## 环境准备

* **处理器：FT2000+**
* **操作系统：麒麟v10**
    * NAME="Kylin Linux Advanced Server"
    * VERSION="V10 (Azalea)"
    * ID="kylin"
    * ID_LIKE="fedora"
    * VARIANT="Server"
    * VARIANT_ID="server"
    * VERSION_ID="V10"
    * PRETTY_NAME="Kylin Linux Advanced Server V10 (Azalea)"
    * ANSI_COLOR="0;31"
    * CPE_NAME="cpe:/o:kylin:enterprise_linux:V10:GA:server"
    * HOME_URL="https://www.kylinos.cn/"
    * BUG_REPORT_URL="https://bugzilla.kylinos.cn/bugzilla/"
    * KYLIN_BUGZILLA_PRODUCT="Kylin Linux Advanced Server 10"
    * KYLIN_BUGZILLA_PRODUCT_VERSION=V10
    * KYLIN_SUPPORT_PRODUCT="Kylin Linux Advanced Server"
    * KYLIN_SUPPORT_PRODUCT_VERSION="V10"
* **Python 版本 2.7.15+/3.5.1+/3.6/3.7 (64 bit)**
* **pip 或 pip3 版本 9.0.1+ (64 bit)**

## 安装步骤

目前在FT2000+处理器加国产化操作系统（麒麟UOS）上安装Paddle，只支持源码编译的方式，接下来详细介绍各个步骤。

<a name="ft_source"></a>
### **源码编译**

1. Paddle依赖cmake进行编译构建，需要cmake版本>=3.10，如果操作系统提供的源包括了合适版本的cmake，直接安装即可，否则需要[源码安装](https://github.com/Kitware/CMake)

    wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8.tar.gz
    tar -xzf cmake-3.16.8.tar.gz && cd cmake-3.16.8
    ./bootstrap && make && sudo make install

2. Paddle内部使用patchelf来修改动态库的rpath，如果操作系统提供的源包括了patchelf，直接安装即可，否则需要源码安装，请参考[patchelf官方文档](https://github.com/NixOS/patchelf)

    ./bootstrap.sh
    ./configure
    make
    make check
    sudo make install

3. 根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装Python依赖库，在飞腾加国产化操作系统环境中，pip安装可能失败或不能正常工作，主要依赖通过源或源码安装的方式安装依赖库，建议使用系统提供源的方式安装依赖库。

4. 将Paddle的源代码克隆到当下目录下的Paddle文件夹中，并进入Paddle目录

    - `git clone https://github.com/PaddlePaddle/Paddle.git`

    - `cd Paddle`

5. 切换到较稳定release分支下进行编译：

    `git checkout [分支名]`

    例如：

    `git checkout v1.8.5`

6. 并且请创建并进入一个叫build的目录下：

    `mkdir build && cd build`

7. 链接过程中打开文件数较多，可能超过系统默认限制导致编译出错，设置进程允许打开的最大文件数：

    `ulimit -n 4096`

8. 执行cmake：

    >具体编译选项含义请参见[编译选项表](../Tables.html#Compile)

        For Python2: cmake .. -DPY_VERSION=2 -DPYTHON_EXECUTABLE=`which python2` -DWITH_ARM=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON
        For Python3: cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_ARM=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON

9. 使用以下命令来编译，注意，因为FT2000+为ARM架构，如果不加`TARGET=ARMV8`则会在编译的时候报错。

    `make TARGET=ARMV8 -j$(nproc)`

10. 编译成功后进入`Paddle/build/python/dist`目录下找到生成的`.whl`包。

11. 在当前机器或目标机器安装编译好的`.whl`包：

    `pip install -U（whl包的名字）`或`pip3 install -U（whl包的名字）`

恭喜，至此您已完成PaddlePaddle在FT环境下的编译安装。


## **验证安装**
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

在mobilenetv1和resnet50模型上测试

    wget -O profile.tar https://paddle-cetc15.bj.bcebos.com/profile.tar?authorization=bce-auth-v1/4409a3f3dd76482ab77af112631f01e4/2020-10-09T10:11:53Z/-1/host/786789f3445f498c6a1fd4d9cd3897ac7233700df0c6ae2fd78079eba89bf3fb
    tar xf profile.tar && cd profile
    python resnet.py --model_file ResNet50_inference/model --params_file ResNet50_inference/params
    python mobilenetv1.py --model_file sample/inference/mobilenetv1/model --params_file sample/inference/mobilenetv1/params


## **如何卸载**
请使用以下命令卸载PaddlePaddle：

`pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`
