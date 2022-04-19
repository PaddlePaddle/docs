# Install on Windows via PIP

## Environmental preparation

### 1.1 PREQUISITES

* **Windows 7/8/10 Pro/Enterprise (64bit)**
* **GPU Version support CUDA 10.1/10.2/11.0/11.1/11.2, and only support single GPU**
* **Python version 3.6+/3.7+/3.8+/3.9+(64bit)**
* **pip version 20.2.2 or above (64bit)**

### 1.2 How to check your environment

* Confirm whether the Python version meets the requirements

  * Use the following command to confirm that it is 3.6+/3.7+/3.8+/3.9+

        python --version


* Confirm whether the version of pip meets the requirements. The version of pip is required to be 20.2.2 or above

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
* NCCL, distribution are not supported on windows now



## INSTALLATION

If you installed Python via Homebrew or the Python website, `pip` was installed with it. If you installed Python 3.x, then you will be using the command `pip3`. We will introduce pip installation here.

### Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install [the CPU Version of PaddlePaddle](#cpu)

* If your computer has NVIDIA® GPU, please make sure that the following conditions are met and install [the GPU Version of PaddlePaddle](#gpu)

  * **CUDA toolkit 10.1/10.2 with cuDNN v7.6.5**

  * **CUDA toolkit 11.0 with cuDNN v8.0.2**

  * **CUDA toolkit 11.1 with cuDNN v8.1.1**

  * **CUDA toolkit 11.2 with cuDNN v8.2.1**

  * **GPU CUDA capability over 3.5**

    You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)


## Installation Step

You can choose the following version of PaddlePaddle to start installation:



#### 2.1 <span id="cpu">CPU Version of PaddlePaddle</span>


  ```
  python -m pip install paddlepaddle==2.3.0rc0 -i https://mirror.baidu.com/pypi/simple
  ```



#### 2.2 <span id="gpu">GPU Version of PaddlePaddle</span>


2.2.1 If you are using CUDA 10.1


  ```
  python -m pip install paddlepaddle-gpu==2.3.0rc0.post101 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
  ```


2.2.2 If you are using CUDA 10.2

  ```
  python -m pip install paddlepaddle-gpu==2.3.0rc0.post102 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
  ```


2.2.3 If you are using CUDA 11.0

If your GPU architecture is 6.0, 6.1, 7.0, 7.5, 8.0, 8.6 (Pascal, Volta, Turing, Ampere), please use the following command to install:

  ```
  python -m pip install paddlepaddle-gpu==2.3.0rc0 -i https://mirror.baidu.com/pypi/simple
  ```

If your GPU architecture is 3.5, 3.7, 5.0, 5.2 (Kepler and Maxwell), first use the following command to download the wheel package to the local, and then use `python -m pip install [name].whl` to install locally ([name] is the name of the wheel package):

  ```
  python -m pip download paddlepaddle-gpu==2.3.0rc0 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html --no-index --no-deps
  ```

2.2.4 If you are using CUDA 11.1

  ```
  python -m pip install paddlepaddle-gpu==2.3.0rc0.post111 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
  ```


2.2.5 If you are using CUDA 11.2

  ```
  python -m pip install paddlepaddle-gpu==2.3.0rc0.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
  ```

Note：

* If you are using ampere-based GPU, CUDA 11 above version is recommended; otherwise CUDA 10.2 is recommended for better performance.

* Please confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python. Depending on the environment, you may need to replace Python in all command lines in the instructions with specific Python path.

* The above commands install the `avx` package by default. If your machine does not support `avx`, you need to install the Paddle package of `noavx`, you can use the following command to install，noavx version paddle wheel only support python3.8：

  First use the following command to download the wheel package to the local, and then use `python -m pip install [name].whl` to install locally ([name] is the name of the wheel package):

   * cpu and mkl version installed on noavx machine：

   ```
   python -m pip download paddlepaddle==2.3.0rc0 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/noavx/stable.html --no-index --no-deps
   ```

   * cpu and openblas version installed on noavx machine：

   ```
   python -m pip download paddlepaddle==2.3.0rc0 -f https://www.paddlepaddle.org.cn/whl/windows/openblas/noavx/stable.html --no-index --no-deps
   ```

   * GPU cuda10.1 version install on noavx machine：

   ```
   python -m pip download paddlepaddle-gpu==2.3.0rc0.post101 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/noavx/stable.html --no-index --no-deps
   ```

   * GPU cuda10.2 version install on noavx machine：

   ```
   python -m pip download paddlepaddle-gpu==2.3.0rc0.post102 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/noavx/stable.html --no-index --no-deps
   ```

   To determine whether your machine supports `avx`, you can install the [CPU-Z](https://www.cpuid.com/softwares/cpu-z.html) tool to view the "processor-instruction set".


* If you want to install the Paddle package with `avx` and `openblas`, you can use the following command to download the wheel package to the local, and then use `python -m pip install [name].whl` to install locally ([name] is the name of the wheel package):

  ```
  python -m pip download paddlepaddle==2.3.0rc0 -f https://www.paddlepaddle.org.cn/whl/windows/openblas/avx/stable.html --no-index --no-deps
  ```

## Verify installation

After the installation is complete, you can use `python` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

* **CPU version of PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU version of PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
