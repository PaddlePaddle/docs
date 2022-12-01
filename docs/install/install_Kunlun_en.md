# Paddle installation for machines with Kunlun XPU card

Paddle supports training and native inference on Kunlun XPU cards.  The latest version is 2.1. Installation methods are given below:

## Installation Method 1：Pre-built Wheel Package

Install the pre-built wheel package that supports Kunlun XPU. At present, Wheel packages are available in the following environments. For any other environment, please choose Installation Method 2 installation from source code compilation.

#### Download the pre-built installation package

**Intel CPU+Kunlun XPU+CentOS 7**

The recommended Linux distribution is CentOS 7.

For Python3.7

```
wget https://paddle-wheel.bj.bcebos.com/kunlun/paddlepaddle-2.1.0-cp37-cp37m-linux_x86_64.whl
```

```
python3.7 -m pip install -U paddlepaddle-2.1.0-cp37-cp37m-linux_x86_64.whl
```

For Python3.6

```
wget https://paddle-wheel.bj.bcebos.com/kunlun/paddlepaddle-2.1.0-cp36-cp36m-linux_x86_64.whl
```

```
python3.6 -m pip install -U ``paddlepaddle-2.1.0-cp36-cp36m-linux_x86_64.whl
```

**Phytium CPU+Kunlun XPU+Kylin release V10**

To obtain the wheel package supporting this system environment, please contact us via official email: Paddle-better@baidu.com

#### Verify installation

After installation, you can use python or python3 to enter the python interpreter, enter:

```import paddle ```

then input:

``` paddle.utils.run_check()```

If "PaddlePaddle is installed successfully!" appears, you have successfully installed it.


#### Training example


**Run a resnet50 training example**

Download the model :

```
cd path_to_clone_PaddleClas
git clone -b release/static https://github.com/PaddlePaddle/PaddleClas.git
```

Download from PaddleClas [github repo](https://github.com/PaddlePaddle/PaddleClas/tree/release/static) is also supported.

Here is the command to start training task:

```
# FLAGS to set the number of Kunlun XPU. Two cards are selected here：
export FLAGS_selected_xpus=0,1
# start training
python3.7 tools/static/train.py -c configs/quick_start/ResNet50_vd_finetune_kunlun.yaml -o use_gpu=False -o use_xpu=True -o is_distributed=False
```


#### How to uninstall

Please use the following command to uninstall PaddlePaddle:

 ```pip uninstall paddlepaddle```

or

 ```pip3 uninstall paddlepaddle ```

In addition, if there are environmental problems with the pre-built wheel package, it is recommended to use Installation Method 2 to compile a package.



## Installation Method 2：Paddle Source Code Compilation

#### Environment preparation

**Intel CPU+Kunlun XPU+CentOS 7**

- **CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz**
- **OS version: CentOS 7.8.2003（ or other CentOS 7 versions）**
- **Python version: 3.6/3.7 (64 bit)**
- **pip or pip3 version: 9.0.1+ (64 bit)**
- **cmake version: 3.15+**
- **gcc/g++ version: 8.2+**

**Phytium CPU+Kunlun XPU+Kylin release V10**

- **CPU: Phytium,FT-2000+/64**
- **OS version: Kylin release V10 (SP1)/(Tercel)-aarch64-Build04/20200711**
- **Python version: 3.6/3.7 (64 bit)**
- **pip 或 pip3 version:  9.0.1+ (64 bit)**
- **cmake version: 3.15+**
- **gcc/g++ version: 8.2+**

#### Compilation and Installation Steps

1. Paddle relies on cmake(version>=3.15) to manage compilation and build. If the source provided by the operating system includes the appropriate version of cmake, you can move to next step, otherwise refer to:

```
wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8.tar.gz
tar -xzf cmake-3.16.8.tar.gz && cd cmake-3.16.8
./bootstrap && make && sudo make install
```

2. Paddle uses patchelf to modify the rpath of the dynamic library. If the source provided by the operating system includes patchelf, you can install it directly, otherwise source installation is required, please refer to:

```
./bootstrap.sh
./configure
make
make check
sudo make install
```

3. Install Python dependency libraries according to [requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt).

4. Clone the source code of Paddle to folder "Paddle" in the current directory, and enter the directory.

```
git clone https://github.com/PaddlePaddle/Paddle.gitcd Paddle
```

Switch to a stable release branch for compilation, Paddle release/2.1 is suggtested：

```git checkout release/2.1```

5. Before compilation, create and enter a new directory called "build":

```mkdir build && cd build```

For specific compilation options, please refer to [Compilation Options Table](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/Tables.html#Compile)

**Intel CPU+Kunlun XPU+CentOS**
During the linking process, too many files opened may exceed the system default limit and cause compilation errors. It is good to set the maximum number of opened files:

```ulimit -n 2048```

Execute cmake ：

For Python3

```
cmake .. -DPY_VERSION=3.6 \
         -DCMAKE_BUILD_TYPE=Release \
         -DWITH_GPU=OFF \
         -DWITH_XPU=ON \
         -DON_INFER=ON \
         -DWITH_PYTHON=ON \
         -DWITH_AVX=ON \
         -DWITH_MKL=ON \
         -DWITH_MKLDNN=ON \
         -DWITH_XPU_BKCL=ON \
         -DWITH_DISTRIBUTE=ON \
         -DWITH_NCCL=OFF

make -j20
```


**Phytium CPU+Kunlun XPU+Kylin release V10**

The XPU SDK shoud be downloaded first, please use the following command:

```
wget https://paddle-wheel.bj.bcebos.com/kunlun/xpu_sdk_v2.0.0.61.tar.gztar xvf xpu_sdk_v2.0.0.61.tar.gzmv output xpu_sdk_v2.0.0.61 xpu_sdk
```

Execute cmake：

```
ulimit -n 4096
python_exe="/usr/bin/python3.7"
export XPU_SDK_ROOT=$PWD/xpu_sdk

cmake .. -DPY_VERSION=3.7 \
         -DPYTHON_EXECUTABLE=$python_exe \
         -DWITH_ARM=ON \
         -DWITH_AARCH64=ON \
         -DWITH_TESTING=OFF \
         -DCMAKE_BUILD_TYPE=Release \
         -DON_INFER=ON \
         -DWITH_XBYAK=OFF \
         -DWITH_XPU=ON \
         -DWITH_GPU=OFF \
         -DWITH_LITE=ON \
         -DLITE_GIT_TAG=release/v2.9 \
         -DXPU_SDK_ROOT=${XPU_SDK_ROOT}

make VERBOSE=1 TARGET=ARMV8 -j32
```

6. After successful compilation, enter the directory "Paddle/build/python/dist" and find the generated .whl package.

7. Copy the generated .whl package to the target machine with Kunlun XPU, and install Python dependency libraries according to [requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt ).  Skip this step if on the target machine already.

8. Install the compiled .whl package on the machine with Kunlun XPU card: pip install -U (the name of the whl package) or pip3 install -U (the name of the whl package).

   Congratulations! So far you have completed the compilation of PaddlePaddle on the Kunlun XPU machine installation.



#### Verify installation

After installation, you can use python or python3 to enter the python interpreter, enter:

```import paddle ```

then input:

``` paddle.utils.run_check()```

If "PaddlePaddle is installed successfully!" appears, you have successfully installed it.



#### Training example

**Run a resnet50 training example**

Download the model :

```
cd path_to_clone_PaddleClasgit clone -b release/static https://github.com/PaddlePaddle/PaddleClas.git
```

Download from PaddleClas [github repo](https://github.com/PaddlePaddle/PaddleClas/tree/release/static) is also supported.

Here is the command to run the training task:

```
# FLAGS to set the number of Kunlun XPU. Two cards are selected here：export FLAGS_selected_xpus=0,1# start trainingpython3.7 tools/static/train.py -c configs/quick_start/ResNet50_vd_finetune_kunlun.yaml -o use_gpu=False -o use_xpu=True -o is_distributed=False
```


#### How to uninstall

Please use the following command to uninstall PaddlePaddle:

 ```pip uninstall paddlepaddle```

or

 ```pip3 uninstall paddlepaddle ```
