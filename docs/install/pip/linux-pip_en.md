# Install on Linux via PIP

## Environmental preparation

### 1.1 PREQUISITES

* **Linux Version (64 bit)**
  * **CentOS 7 (GPUVersion Supports CUDA 10.1/10.2/11.2**)**
  * **Ubuntu 16.04 (GPUVersion Supports CUDA 10.1/10.2/11.2)**
  * **Ubuntu 18.04 (GPUVersion Supports CUDA 10.1/10.2/11.2)**

* **Python Version: 3.7/3.8/3.9/3.10 (64 bit)**

* **pip or pip3 Version 20.2.2 or above (64 bit)**

### 1.2 How to check your environment

* You can use the following commands to view the local operating system and bit information

  ```
  uname -m && cat /etc/*release
  ```



* Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

  * Use the following command to output Python path. Depending on the environment, you may need to replace Python in all command lines in the description with specific Python path

    ```
    which python
    ```



* You need to confirm whether the version of Python meets the requirements

  * Use the following command to confirm that it is 3.7/3.8/3.9/3.10

        python --version

* It is required to confirm whether the version of pip meets the requirements. The version of pip is required to be 20.2.2 or above


    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```



* You need to confirm that Python and pip are 64bit, and the processor architecture is x86_64(or called x64、Intel 64、AMD64). Currently, paddlepaddle does not support arm64 architecture. The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64"

    ```
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```



* The installation package provided by default requires computer support for MKL

* If you do not know the machine environment, please download and use[Quick install script](https://fast-install.bj.bcebos.com/fast_install.sh), for instructions please refer to[here](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/install_script.md)。



## INSTALLATION

If you installed Python via Homebrew or the Python website, `pip` was installed with it. If you installed Python 3.x, then you will be using the command `pip3`.

### Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install [the CPU Version of PaddlePaddle](#cpu)

* If your computer has NVIDIA® GPU, please make sure that the following conditions are met and install [the GPU Version of PaddlePaddle](#gpu)

  * **CUDA toolkit 10.1/10.2 with cuDNN v7.6+(for multi card support, NCCL2.7 or higher)**

  * **CUDA toolkit 11.2 with cuDNN v8.1.1(for multi card support, NCCL2.7 or higher)**

  * **Hardware devices with GPU computing power over 3.5**

    You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* If you need to use a multi-card environment, please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here are the installation instructions of nccl2 under CUDA9 and cuDNN7. For more version installation information, please refer to NVIDIA [Official Website](https://developer.nvidia.com/nccl)):

  * **CentOS system can refer to the following commands**

        wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm

    ```
    rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
    ```

    ```
    yum update -y
    ```

    ```
    yum install -y libnccl-2.3.7-2+cuda9.0 libnccl-devel-2.3.7-2+cuda9.0 libnccl-static-2.3.7-2+cuda9.0
    ```

  * **Ubuntu system can refer to the following commands**

    ```
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    ```

    ```
    dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    ```

    ```
    sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0
    ```



## Installation Step

You can choose the following version of PaddlePaddle to start installation:



#### 2.1 CPU Versoion of PaddlePaddle


  ```
  python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
  ```



#### 2.2 GPU Version of PaddlePaddle



2.2.1 If you are using CUDA 10.1


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```



2.2.2 If you are using CUDA 10.2


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```


2.2.3 If you are using CUDA 11.0


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```


2.2.4 If you are using CUDA 11.1


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post111 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```



2.2.5 If you are using CUDA 11.2


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```



Note：

* If you are using ampere-based GPU, CUDA 11.2 is recommended; otherwise CUDA 10.2 is recommended for better performance. please refer to: [GPU architecture comparison table](https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables.html#nvidia-gpu)

* Please confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python. Depending on the environment, you may need to replace Python in all command lines in the instructions with Python 3 or specific Python path.



## Verify installation

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

- ***CPU version of PaddlePaddle\***: `python -m pip uninstall paddlepaddle`
- ***GPU version of PaddlePaddle\***: `python -m pip uninstall paddlepaddle-gpu`
