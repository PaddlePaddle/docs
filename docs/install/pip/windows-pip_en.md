# Install on Windows via PIP

## Environmental preparation

### 1.1 How to check your environment

* Confirm whether the Python version meets the requirements

  * Use the following command to confirm that it is 3.8+/3.9+/3.10+/3.11+/3.12+

        python --version


* Confirm whether the version of pip meets the requirements. The version of pip is required to be 20.2.2 or above

    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```

* You need to confirm that Python and pip are 64bit, and the processor architecture is x86_64(or called x64、Intel 64、AMD64). The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64"

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

  * **CUDA toolkit 11.2 with cuDNN v8.2.1(for PaddleTensorRT deployment, TensorRT8.2.4.2)**

  * **CUDA toolkit 11.6 with cuDNN v8.4.0(for PaddleTensorRT deployment, TensorRT8.4.0.6)**

  * **CUDA toolkit 11.7 with cuDNN v8.4.1(for PaddleTensorRT deployment, TensorRT8.4.2.4)**

  * **CUDA toolkit 11.8 with cuDNN v8.6.0(for PaddleTensorRT deployment, TensorRT8.5.1.7)**

  * **CUDA toolkit 12.0 with cuDNN v8.9.1(for multi card support, NCCL2.7 or higher；for PaddleTensorRT deployment, TensorRT8.6.1.6)**

  * **GPU CUDA capability over 3.5**

    You can refer to NVIDIA official documents for installation process and configuration method of CUDA, cuDNN and TensorRT. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，[TensorRT](https://developer.nvidia.com/tensorrt)


## Installation Step

You can choose the following version of PaddlePaddle to start installation:



#### 2.1 <span id="cpu">CPU Version of PaddlePaddle</span>


  ```
  python -m pip install paddlepaddle==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```



#### 2.2 <span id="gpu">GPU Version of PaddlePaddle</span>



2.2.1 If you are using CUDA 11.2

  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post112 -i https://www.paddlepaddle.org.cn/packages/stable/cu112/
  ```

2.2.2 If you are using CUDA 11.6

  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post116 -i https://www.paddlepaddle.org.cn/packages/stable/cu116/
  ```

2.2.3 If you are using CUDA 11.7

  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post117 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/
  ```

2.2.4 If you are using CUDA 11.8

  ```
  python -m pip install paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

2.2.5 If you are using CUDA 12.0

  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
  ```


## Verify installation

After the installation is complete, you can use `python` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

* **CPU version of PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU version of PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
