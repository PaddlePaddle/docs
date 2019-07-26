# **Installation on Windows**

## Operating Environment

* *Windows 7/8/10 Pro/Enterprise(64bit)(CUDA 8.0/9.0 are supported, and only single GPU is supported)*
* *Python 2.7.15+/3.5.1+/3.6/3.7(64bit)*
* *pip or pip3 9.0.1+(64bit)*

### Precautions

* The default installation package requires your computer to support AVX instruction set and MKL. If your environment doesn’t support AVX instruction set and MKL, please download [these](./Tables.html/#ciwhls-release) `no-avx`, `openblas` versions of installation package.
* The current version doesn’t support functions related to NCCL and distributed learning.

## CPU or GPU

* If your computer doesn’t have NVIDIA® GPU, please install the CPU version of PaddlePaddle

* If your computer has NVIDIA® GPU, and it satisfies the following requirements, we recommend you to install the GPU version of PaddlePaddle
    * *CUDA Toolkit 8.0/9.0 with cuDNN v7.3+*
    * *GPU's computing capability exceeds 1.0*

Please refer to the NVIDIA official documents for the installation process and the configuration methods of [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/).

## Installation Method

There are 3 ways to install PaddlePaddle on Windows:

* pip installation (recommended)
* [Docker installation](./install_Docker.html)
* [source code compilation and installation](./compile/compile_Windows.html/#win_source)

We would like to introduce the pip installation here.

## Installation Process

* CPU version of PaddlePaddle: `pip install paddlepaddle` or `pip3 install paddlepaddle`
* GPU version of PaddlePaddle: `pip install paddlepaddle-gpu` or `pip3 install paddlepaddle-gpu`

There is a checking function below for [verifyig whether the installation is successful](#check). If you have any further questions, please check the [FAQ part](./FAQ.html).

Notice:

* The version of pip and the version of python should be corresponding: python2.7 corresponds to `pip`; python3.x corresponds to `pip3`.
* `pip install paddlepaddle-gpu` This command will install PaddlePaddle that supports CUDA 8.0/9.0 cuDNN v7.3+, Currently, PaddlePaddle doesn't support any other version of CUDA or cuDNN on Windows.

<a name="check"></a>
## Installation Verification
After completing the installation process, you can use `python` or `python3` to enter python interpreter and input `import paddle.fluid as fluid` and then `fluid.install_check.run_check()` to check whether the installation is successful.

If you see `Your Paddle Fluid is installed succesfully!`, your installation is verified successful.

## Uninstall PaddlePaddle

* ***CPU version of PaddlePaddle***: `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`

* ***GPU version of PaddlePaddle***: `pip uninstall paddlepaddle-gpu` or `pip3 uninstall paddlepaddle-gpu`
