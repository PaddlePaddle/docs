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

  * Use the following command to confirm that it is 3.7/3.8/3.9/3.10

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

  * **CUDA toolkit 10.2 with cuDNN v7.6.5(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT7.0.0.11)**

  * **CUDA toolkit 11.2 with cuDNN v8.2.1(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.0.3.4)**

  * **CUDA toolkit 11.6 with cuDNN v8.4.0(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.4.0.6)**

  * **CUDA toolkit 11.7 with cuDNN v8.4.1(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.4.2.4)**

  * **CUDA toolkit 11.8 with cuDNN v8.6.0(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.5.1.7)**

  * **Hardware devices with GPU computing power over 3.5**

    You can refer to NVIDIA official documents for installation process and configuration method of CUDA, cuDNN and TensorRT. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，[TensorRT](https://developer.nvidia.com/tensorrt)

* If you need to use a multi-card environment, please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here are the installation instructions of nccl2 under CUDA10.2 and cuDNN7. For more version installation information, please refer to NVIDIA [Official Website](https://developer.nvidia.com/nccl)):

  * **Centos system can refer to the following commands**

        wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm

    ```
    rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
    ```

    ```
    yum update -y
    ```

    ```
    yum install -y libnccl-2.7.8-1+cuda10.2 libnccl-devel-2.7.8-1+cuda10.2 libnccl-static-2.7.8-1+cuda10.2
    ```

  * **Ubuntu system can refer to the following commands**

    ```
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    ```

    ```
    dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    ```

    ```
    sudo apt install -y libnccl2=2.7.8-1+cuda10.2 libnccl-dev=2.7.8-1+cuda10.2
    ```



## Installation Step

You can choose the following version of PaddlePaddle to start installation:



#### 2.1 <span id="cpu">CPU Version of PaddlePaddle</span>


  ```
  python3 -m pip install paddlepaddle==2.5.0rc1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```



#### 2.2 <span id="gpu">GPU Version of PaddlePaddle</span>



2.2.1 If you are using CUDA 10.2


  ```
  python3 -m pip install paddlepaddle-gpu==2.5.0rc1.post102 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
  ```


  CUDA10.2 with cuDNN dynamic library PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.5.0rc1.post102 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html
  ```


2.2.2 If you are using CUDA 11.2


  ```
  python3 -m pip install paddlepaddle-gpu==2.5.0rc1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```


  CUDA11.2 with cuDNN dynamic library PaddlePaddle, you can use the following command to download the wheel package to the local, and then use `python3 -m pip install [name].whl` to install locally ([name] is the name of the wheel package)


  ```
  python3 -m pip download paddlepaddle-gpu==2.5.0rc1 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html  --no-index --no-deps
  ```


2.2.3 If you are using CUDA 11.6


  ```
  python3 -m pip install paddlepaddle-gpu==2.5.0rc1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
  ```


  CUDA11.6 with cuDNN dynamic library PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.5.0rc1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html
  ```


2.2.4 If you are using CUDA 11.7


  ```
  python3 -m pip install paddlepaddle-gpu==2.5.0rc1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
  ```


  CUDA11.7 with cuDNN dynamic library PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.5.0rc1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html
  ```


2.2.5 If you are using CUDA 11.8


  ```
  python3 -m pip install paddlepaddle-gpu==2.5.0rc1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
  ```


  CUDA11.8 with cuDNN dynamic library PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.5.0rc1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html
  ```


Note：

* If you are using ampere-based GPU, CUDA 11 above version is recommended; otherwise CUDA 10.2 is recommended for better performance.

* Please confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python. Depending on the environment, you may need to replace python3 in all command lines in the instructions with specific Python path.

* If you want to use the tsinghua pypi, you can use the following command:

  ```
   python3 -m pip install paddlepaddle-gpu==[Version] -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

* The above commands install the `avx` and `mkl` package by default. Paddle no longer supports `noavx` package. To determine whether your machine supports `avx`, you can use the following command. If the output contains `avx`, it means that the machine supports `avx`:
   ```
   cat /proc/cpuinfo | grep -i avx
   ```

* If you want to install the Paddle package with `avx` and `openblas`, you can use the following command to download the wheel package to the local, and then use `python3 -m pip install [name].whl` to install locally ([name] is the name of the wheel package):

  ```
  python3 -m pip download paddlepaddle==2.5.0rc1 -f https://www.paddlepaddle.org.cn/whl/linux/openblas/avx/stable.html --no-index --no-deps
  ```



## Verify installation

After the installation is complete, you can use `python3` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

- **CPU version of PaddlePaddle**: `python3 -m pip uninstall paddlepaddle`
- **GPU version of PaddlePaddle**: `python3 -m pip uninstall paddlepaddle-gpu`
