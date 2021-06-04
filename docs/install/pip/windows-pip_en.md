# Install on Windows via PIP

## Environmental preparation

### 1.1 PREQUISITES

* **Windows 7/8/10 Pro/Enterprise (64bit)**
  * **GPU version support CUDA 10.1/10.2/11.2，only support single card**

* **Python version 3.6+/3.7+/3.8+/3.9+ (64 bit)**

* **pip version 20.2.2 or above (64 bit)**

### 1.2 How to check your environment

* Confirm the local operating system and bit information


* Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

  ```
  where python
  ```



* You need to confirm whether the version of Python meets the requirements

  * Use the following command to confirm that it is 3.6/3.7/3.8/3.9

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

* If you do not know the machine environment, please download and use[Quick install script](https://fast-install.bj.bcebos.com/fast_install.sh), for instructions please refer to[here](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## INSTALLATION

If you installed Python via Homebrew or the Python website, `pip` was installed with it. If you installed Python 3.x, then you will be using the command `pip3`. We will introduce pip installation here.

### Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install [the CPU Version of PaddlePaddle](#cpu)

* If your computer has NVIDIA® GPU, please make sure that the following conditions are met and install [the GPU Version of PaddlePaddle](#gpu)

  * **CUDA toolkit 10.1/10.2 with cuDNN v7.6.5+**

  * **CUDA toolkit 11.2 with cuDNN v8.1.1**

  * **Hardware devices with GPU computing power over 3.0**

    You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)


## Installation Step

You can choose the following version of PaddlePaddle to start installation:




#### 2.1 CPU Versoion of PaddlePaddle


  ```
  python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```



#### 2.2 GPU Version of PaddlePaddle


2.2.1 CUDA10.1的PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==2.1.0.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
  ```


2.2.2 CUDA10.2的PaddlePaddle

  ```
  python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```


2.2.3 CUDA11.2的PaddlePaddle

  ```
  python -m pip install paddlepaddle-gpu==2.1.0.post112 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
  ```

Note：

* If you are using ampere-based GPU, CUDA 11.2 is recommended; otherwise CUDA 10.2 is recommended for better performance.

* Please confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python. Depending on the environment, you may need to replace Python in all command lines in the instructions with specific Python path.

* The above commands install the `avx` package by default. If your machine does not support `avx`, you need to install the Paddle package of `noavx`, you can use the following command to install，noavx version paddle wheel only support python3.8：

   * cpu and mkl version installed on noavx machine：

   ```
   python -m pip install paddlepaddle==2.1.0 -f http://www.paddlepaddle.org.cn/whl/mkl/stable/noavx/html --no-index
   ```

   * cpu and openblas version installed on noavx machine：

   ```
   python -m pip install paddlepaddle==2.1.0 -f https://www.paddlepaddle.org.cn/whl/openblas/stable/noavx.html --no-index
   ```

   * GPU cuda10.1 version install on noavx machine：：

   ```
   python -m pip install paddlepaddle-gpu==2.1.0.post101 -f https://www.paddlepaddle.org.cn/whl/mkl/stable/noavx.html
   ```

   * GPU cuda10.2 version install on noavx machine：：

   ```
   python -m pip install paddlepaddle-gpu==2.1.0 -f https://www.paddlepaddle.org.cn/whl/mkl/stable/noavx.html --no-index
   ```

   To determine whether your machine supports `avx`, you can use the following command. If the output contains `avx`, it means that the machine supports `avx`:
   ```
   cat /proc/cpuinfo | grep -i avx
   ```


* If you want to install the Paddle package built with `tensorrt`, please use the following command:

  ```
  python -m pip install paddlepaddle-gpu==[版本号] -f https://paddlepaddle.org.cn/whl/stable/tensorrt.html
  ```


## Verify installation

After the installation is complete, you can use `python` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

* **CPU version of PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU version of PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
