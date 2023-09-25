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
* If you are in a WSL2 environment, it is recommended to install using Paddle according to the Linux method


## INSTALLATION

If you installed Python via Homebrew or the Python website, `pip` was installed with it. If you installed Python 3.x, then you will be using the command `pip3`. We will introduce pip installation here.

### Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install [the CPU Version of PaddlePaddle](#cpu)

* If your computer has NVIDIA® GPU, please make sure that the following conditions are met and install [the GPU Version of PaddlePaddle](#gpu)

  * **CUDA toolkit 10.1/10.2 with cuDNN v7.6.5+**

  * **CUDA toolkit 11.0 with cuDNN v8.0.2**

  * **CUDA toolkit 11.1 with cuDNN v8.1.1**

  * **CUDA toolkit 11.2 with cuDNN v8.2.1**

  * **GPU CUDA capability over 3.5**

    You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)


## Installation Step

You can choose the following version of PaddlePaddle to start installation:


#### 2.1 CPU Versoion of PaddlePaddle


  ```
  python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/windows/cpu-mkl-avx/develop.html
  ```



#### 2.2 GPU Version of PaddlePaddle


2.2.1 If you are using CUDA 10.1


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```


2.2.2 If you are using CUDA 10.2

  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```

2.2.3 If you are using CUDA 11.0


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post110 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```


2.2.4 If you are using CUDA 11.1


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post111 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```


2.2.5 If you are using CUDA 11.2

  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```

Note：

* If you are using ampere-based GPU, CUDA 11.2 is recommended; otherwise CUDA 10.2 is recommended for better performance. please refer to: [GPU architecture comparison table](https://www.paddlepaddle.org.cn/documentation/docs/en/install/Tables.html#nvidia-gpu)

* Please confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python. Depending on the environment, you may need to replace Python in all command lines in the instructions with specific Python path.


## Verify installation

After the installation is complete, you can use `python` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

* **CPU version of PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU version of PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
