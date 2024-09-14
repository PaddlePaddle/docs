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

  * **CUDA toolkit 11.2 with cuDNN v8.2.1(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.0.3.4)**

  * **CUDA toolkit 11.6 with cuDNN v8.4.0(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.4.0.6)**

  * **CUDA toolkit 11.7 with cuDNN v8.4.1(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.4.2.4)**

  * **CUDA toolkit 11.8 with cuDNN v8.6.0(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.5.1.7)**

  * **CUDA toolkit 12.0 with cuDNN v8.9.1(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.6.1.6)**

  * **Hardware devices with GPU computing power over 3.5**

    You can refer to NVIDIA official documents for installation process and configuration method of CUDA, cuDNN and TensorRT. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，[TensorRT](https://developer.nvidia.com/tensorrt)

* If you need to use a multi-card environment, please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here are the installation instructions of nccl2 under CUDA11.2 and cuDNN7. For more version installation information, please refer to NVIDIA [Official Website](https://developer.nvidia.com/nccl)):


    ```
    rm -f /usr/local/lib/libnccl.so
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-2.10.3-1+cuda11.4.x86_64.rpm
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-devel-2.10.3-1+cuda11.4.x86_64.rpm
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-static-2.10.3-1+cuda11.4.x86_64.rpm
    rpm -ivh libnccl-2.10.3-1+cuda11.4.x86_64.rpm
    rpm -ivh libnccl-devel-2.10.3-1+cuda11.4.x86_64.rpm
    rpm -ivh libnccl-static-2.10.3-1+cuda11.4.x86_64.rpm
    ```



## Installation Step

You can choose the following version of PaddlePaddle to start installation:



#### 2.1 <span id="cpu">CPU Version of PaddlePaddle</span>


  ```
  python3 -m pip install paddlepaddle==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```



#### 2.2 <span id="gpu">GPU Version of PaddlePaddle</span>



2.2.1 If you are using CUDA 11.2


  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post112 -i https://www.paddlepaddle.org.cn/packages/stable/cu112/
  ```


  CUDA11.2 with cuDNN dynamic library PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.6.2.post112 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/
  ```


2.2.2 If you are using CUDA 11.6


  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post116 -i https://www.paddlepaddle.org.cn/packages/stable/cu116/
  ```


  CUDA11.6 with cuDNN dynamic library PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post116 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/
  ```


2.2.3 If you are using CUDA 11.7


  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post117 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/
  ```


  CUDA11.7 with cuDNN dynamic library PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.6.2.post117 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/
  ```


2.2.4 If you are using CUDA 11.8


  ```
  python3 -m pip install paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```


  CUDA11.8 with cuDNN dynamic library PaddlePaddle


  ```
  python3 -m pip download paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/

  ```


2.2.5 If you are using CUDA 12.0


  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
  ```


  CUDA12.0 with cuDNN dynamic library PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/
  ```


## Verify installation

After the installation is complete, you can use `python3` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

- **CPU version of PaddlePaddle**: `python3 -m pip uninstall paddlepaddle`
- **GPU version of PaddlePaddle**: `python3 -m pip uninstall paddlepaddle-gpu`
