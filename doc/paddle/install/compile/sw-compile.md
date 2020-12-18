# **申威下从源码编译**

## 环境准备

* **处理器：SW6A**
* **操作系统：普华, iSoft Linux 5**
* **Python 版本 2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**
* **pip 或 pip3 版本 9.0.1+ (64 bit)**

申威机器为SW架构，目前生态支持的软件比较有限，本文以比较trick的方式在申威机器上源码编译Paddle，未来会随着申威软件的完善不断更新。

## 安装步骤

本文在申威处理器下安装Paddle，接下来详细介绍各个步骤。

<a name="sw_source"></a>
### **源码编译**

1. 将Paddle的源代码克隆到当下目录下的Paddle文件夹中，并进入Paddle目录

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    ```

    ```
    cd Paddle
    ```

2. 切换到`develop`分支下进行编译：

    ```
    git checkout develop
    ```

3. Paddle依赖cmake进行编译构建，需要cmake版本>=3.10，检查操作系统源提供cmake的版本，使用源的方式直接安装cmake, `apt install cmake`, 检查cmake版本, `cmake --version`, 如果cmake >= 3.10则不需要额外的操作，否则请修改Paddle主目录的`CMakeLists.txt`, `cmake_minimum_required(VERSION 3.10)` 修改为 `cmake_minimum_required(VERSION 3.0)`.

4. 由于申威暂不支持openblas，所以在此使用blas + cblas的方式，在此需要源码编译blas和cblas。

    ```
    pushd /opt
    wget http://www.netlib.org/blas/blas-3.8.0.tgz
    wget http://www.netlib.org/blas/blast-forum/cblas.tgz
    tar xzf blas-3.8.0.tgz
    tar xzf cblas.tgz
    pushd BLAS-3.8.0
    make
    popd
    pushd CBLAS
    # 修改Makefile.in中BLLIB为BLAS-3.8.0的编译产物blas_LINUX.a
    make
    pushd lib
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD
    ln -s cblas_LINUX.a libcblas.a
    cp ../../BLAS-3.8.0/blas_LINUX.a .
    ln -s blas_LINUX.a libblas.a
    popd
    popd
    popd
    ```

5. 根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装Python依赖库，注意在申威系统中一般无法直接使用pip或源码编译安装python依赖包，建议使用源的方式安装，如果遇到部分依赖包无法安装的情况，请联系操作系统服务商提供支持。此外也可以通过pip安装的时候加--no-deps的方式来避免依赖包的安装，但该种方式可能导致包由于缺少依赖不可用。

6. 请创建并进入一个叫build的目录下：

    ```
    mkdir build && cd build
    ```

7. 链接过程中打开文件数较多，可能超过系统默认限制导致编译出错，设置进程允许打开的最大文件数：

    ```
    ulimit -n 4096
    ```

8. 执行cmake：

    >具体编译选项含义请参见[编译选项表](../Tables.html#Compile)

    ```
    CBLAS_ROOT=/opt/CBLAS
    ```

    For Python2:
    ```
    cmake .. -DPY_VERSION=2 -DPYTHON_EXECUTABLE=`which python2` -DWITH_MKL=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_PYTHON=ON -DREFERENCE_CBLAS_ROOT=${CBLAS_ROOT} -DWITH_CRYPTO=OFF -DWITH_XBYAK=OFF -DWITH_SW=ON -DCMAKE_CXX_FLAGS="-Wno-error -w"
    ```
    For Python3:
    ```
    cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MKL=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_PYTHON=ON -DREFERENCE_CBLAS_ROOT=${CBLAS_ROOT} -DWITH_CRYPTO=OFF -DWITH_XBYAK=OFF -DWITH_SW=ON -DCMAKE_CXX_FLAGS="-Wno-error -w"
    ```

9. 编译。

    ```
    make -j$(nproc)
    ```

10. 编译成功后进入`Paddle/build/python/dist`目录下找到生成的`.whl`包。

11. 在当前机器或目标机器安装编译好的`.whl`包：

    ```
    python2 -m pip install -U（whl包的名字）`或`python3 -m pip install -U（whl包的名字）
    ```

恭喜，至此您已完成PaddlePaddle在FT环境下的编译安装。

## **验证安装**
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

在mobilenetv1和resnet50模型上测试

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
请使用以下命令卸载PaddlePaddle：

```
python3 -m pip uninstall paddlepaddle
```
或
```
python3 -m pip uninstall paddlepaddle
```

## **备注**

已在申威下测试过resnet50, mobilenetv1, ernie， ELMo等模型，基本保证了预测使用算子的正确性，但可能会遇到浮点异常的问题，该问题我们后续会和申威一起解决，如果您在使用过程中遇到计算结果错误，编译失败等问题，请到[issue](https://github.com/PaddlePaddle/Paddle/issues)中留言，我们会及时解决。

预测文档见[doc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html)，使用示例见[Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)
