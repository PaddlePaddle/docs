# Install on Linux via PIP

## Environmental preparation

### 1.1 How to check your environment

* You can use the following commands to view the local operating system and bit information

  ```
  uname -m && cat /etc/*release
  ```



* Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

  * Use the following command to output Python path. Depending on the environment, you may need to replace Python in all command lines in the description with specific Python path

    ```
    which python3
    ```



* You need to confirm whether the version of Python meets the requirements

  * Use the following command to confirm that it is 3.8/3.9/3.10/3.11/3.12

        python3 --version

* It is required to confirm whether the version of pip meets the requirements. The version of pip is required to be 20.2.2 or above


    ```
    python3 -m ensurepip
    ```

    ```
    python3 -m pip --version
    ```



* You need to confirm that Python and pip are 64bit, and the processor architecture is x86_64(or called x64、Intel 64、AMD64). The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64"

    ```
    python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```



* The installation package provided by default requires computer support for MKL

* If you do not know the machine environment, please download and use[Quick install script](https://fast-install.bj.bcebos.com/fast_install.sh), for instructions please refer to[here](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## INSTALLATION

### Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install [the CPU Version of PaddlePaddle](#cpu)

* If your computer has NVIDIA® GPU, please make sure that the following conditions are met and install [the GPU Version of PaddlePaddle](#gpu)

  * **CUDA toolkit 11.8 with cuDNN v8.6.0(for PaddleTensorRT deployment, TensorRT8.5.1.7)**

  * **CUDA toolkit 12.3 with cuDNN v9.0.0(for PaddleTensorRT deployment, TensorRT8.6.1.6)**

  * **Hardware devices with GPU computing power over 6.0**

    You can refer to NVIDIA official documents for installation process and configuration method of CUDA, cuDNN and TensorRT. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，[TensorRT](https://developer.nvidia.com/tensorrt)



## Installation Step

You can choose the following version of PaddlePaddle to start installation:



#### 2.1 <span id="cpu">CPU Version of PaddlePaddle</span>


  ```
  python3 -m pip install paddlepaddle==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  ```


#### 2.2 <span id="gpu">GPU Version of PaddlePaddle</span>


2.2.4 If you are using CUDA 11.8


  ```
  python3 -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  ```


2.2.5 If you are using CUDA 12.3


  ```
  python3 -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
  ```


Note：

* Please confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python. Depending on the environment, you may need to replace python3 in all command lines in the instructions with specific Python path.

* The above commands install the `avx` and `mkl` package by default. Paddle no longer supports `noavx` package. To determine whether your machine supports `avx`, you can use the following command. If the output contains `avx`, it means that the machine supports `avx`:
   ```
   cat /proc/cpuinfo | grep -i avx
   ```

* If you want to install the Paddle package with `avx` and `openblas`, you can use the following command to download the wheel package to the local, and then use `python3 -m pip install [name].whl` to install locally ([name] is the name of the wheel package):

  ```
  python3 -m pip install https://paddle-wheel.bj.bcebos.com/3.0.0-beta0/linux/linux-cpu-openblas-avx/paddlepaddle-3.0.0b2-cp38-cp38-linux_x86_64.whl
  ```



## Verify installation

After the installation is complete, you can use `python3` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

- **CPU version of PaddlePaddle**: `python3 -m pip uninstall paddlepaddle`
- **GPU version of PaddlePaddle**: `python3 -m pip uninstall paddlepaddle-gpu`
