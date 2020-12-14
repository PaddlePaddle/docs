# Paddle installation for machines with Kunlun XPU card

## Installation Method 1：pre-built Wheel package

Install the pre-built wheel package that supports Kunlun XPU. At present, this wheel package only supports the Ubuntu environment. For other environments, please choose Installation Method 2 to installing from source code compilation.

#### Download the pre-built installation package

Python3.7

o  ```wget https://paddle-wheel.bj.bcebos.com/kunlun/paddlepaddle-2.0.0rc1-cp37-cp37m-linux_x86_64.whl```

o  ```python3.7 -m pip install -U paddlepaddle-2.0.0rc1-cp37-cp37m-linux_x86_64.whl ```

Python2.7

o  ```wget https://paddle-wheel.bj.bcebos.com/kunlun/paddlepaddle-2.0.0rc1-cp27-cp27mu-linux_x86_64.whl```

o  ```python2.7 -m pip install -U paddlepaddle-2.0.0rc1-cp27-cp27mu-linux_x86_64.whl```



#### Verify installation

After installation, you can use python or python3 to enter the python interpreter, enter:

```import paddle.fluid as fluid ```

then input:

``` fluid.install_check.run_check()```

If "Your Paddle Fluid is installed succesfully!" appears, it means you have successfully installed it.



#### Training example

Download and run the example:

o  ```wget https://fleet.bj.bcebos.com/kunlun/mnist_example.py ```

o  ```python mnist_example.py --use_device=xpu --num_epochs=5```

Note: There are three devices of cpu/gpu/xpu in the back end, which can be configured by yourself.


#### How to uninstall

Please use the following command to uninstall PaddlePaddle:

 ```pip uninstall paddlepaddle```

or

 ```pip3 uninstall paddlepaddle ```

In addition, if there are environmental problems with the pre-built wheel package that supports Kunlun XPU card, it is recommended to use Installation Method 2 to compile the package that supports Kunlun XPU.



## Installation Method 2：by compiling Paddle with Kunlun XPU support

#### Environment preparation

- **CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz**
- **OS version：Ubuntu 16.04.6 LTS**
- **Python version 2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**
- **pip or pip3 version 9.0.1+ (64 bit)**
- **cmake version 3.10+**
- **gcc/g++ version 8.2+**

#### install steps

For running Paddle on Kunlun XPU machine, we must first install the Paddle package compiled with Kunlun XPU support.

##### **source code compilation**

1. Paddle relies on cmake(version>=3.10) to manage compilation and build. If the source provided by the operating system includes the appropriate version of cmake, you can install it directly, otherwise refer to:

   o  ```wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8.tar.gz```

   o  ```tar -xzf cmake-3.16.8.tar.gz && cd cmake-3.16.8 ```

   o  ```./bootstrap && make && sudo make install```

2. Paddle uses patchelf to modify the rpath of the dynamic library. If the source provided by the operating system includes patchelf, you can install it directly, otherwise source installation is required, please refer to:

   o  ```./bootstrap.sh ```

   o ``` ./configure ```

   o ``` make ```

   o ``` make check ```

   o  ```sudo make install```

3. Install Python dependency libraries according to [requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt).

4. Clone the source code of Paddle to the Paddle folder in the current directory, and enter the Paddle directory

   o ``` git clone https://github.com/PaddlePaddle/Paddle.git```

   o  ```cd Paddle```

5. Switch to a stable release branch for compilation：

   ```git checkout [分支名]```

   for example：

   ```git checkout release/2.0-rc1```

   Currently only 2.0-rc1 and later branches support Kunlun XPU.

6. Please create and enter a directory called build:

   ```mkdir build && cd build```

7. There are many open files during the linking process, which may exceed the system default limit and cause compilation errors. Set the maximum number of open files in the process:

   ```ulimit -n 4096```

8. Execute cmake ：
9. For the meaning of specific compilation options, please refer to [Compile Options Table](https://www.paddlepaddle.org.cn/install/quick/Tables.html#Compile)

For Python2: ```cmake .. -DPY_VERSION=2 -DPYTHON_EXECUTABLE=`which python2` -DWITH_MKL=OFF -DWITH_XPU=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release ```

For Python3: ```cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MKL=OFF -DWITH_XPU=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release ```

10. Use the following command to compile：

    ```make -j$(nproc)```

11. After successful compilation, enter the Paddle/build/python/dist directory and find the generated .whl package.

12. Copy the generated .whl package to the target machine with Kunlun XPU, and according to [requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt ) Install Python dependency libraries. (If the compiling machine is also a machine with Kunlun XPU card, skip this step)

13. Install the compiled .whl package on the machine with Kunlun XPU card: pip install -U (the name of the whl package) or pip3 install -U (the name of the whl package) Congratulations, so far you have completed the compilation of PaddlePaddle on the Kunlun XPU machine installation.



#### Verify installation

After installation, you can use python or python3 to enter the python interpreter, enter:

```import paddle.fluid as fluid ```

then input:

``` fluid.install_check.run_check()```

If "Your Paddle Fluid is installed succesfully!" appears, it means you have successfully installed it.



#### Training example

Download and run the example:

o  ```wget https://fleet.bj.bcebos.com/kunlun/mnist_example.py ```

o  ```python mnist_example.py --use_device=xpu --num_epochs=5```

Note: There are three devices of cpu/gpu/xpu in the back end, which can be configured by yourself.


#### How to uninstall

Please use the following command to uninstall PaddlePaddle:

 ```pip uninstall paddlepaddle```

or

 ```pip3 uninstall paddlepaddle ```


## Appendix: Current models supported by Kunlun XPU adaptation

|  Models   | Download links  |
|  ----  | ----  |
| ResNet50  | [click to download](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams) |
| MobileNetv3  | [click to download](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams) |
