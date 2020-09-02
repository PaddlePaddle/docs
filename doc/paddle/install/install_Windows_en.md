# **Installation on Windows**

## Environment Preparation

* **Windows 7/8/10 Pro/Enterprise(64bit)**
    * **GPU Version support CUDA 9.0/9.1/9.2/10.0/10.1, and only support single GPU**
* **Python version 2.7.15+/3.5.1+/3.6/3.7(64bit)**
* **pip version 9.0.1+(64bit)**

### Precautions

* Confirm the Python you install PaddlePaddle is the version you expected, because your computer may have more than one python, use the following command:

    python --version

    * If you are using Python 2, the output should be 2.7.15+

    * If you are using Python 3, the output should be 3.5.1+/3.6+/3.7+

* If Python doesn't match your expected version, use the following command to see if Python's path is where you expect it to be:

    where python

    * If you are using Python 2, The installation directory for python2.7 should be on the first line

    * If you are using Python 3, The installation directory for python3.5.1+/3.6+/3.7+ should be on the first line

    * You can adjust it in any of the following ways:

        * Use specific Python paths to execute commands（e.g. C:\Python36\python.exe corresponding to Python 3，C:\Python27\python.exe corresponding to Python 2)  
        * By modifying the environment variable, set your expected installation path in the first order (please modify it in control panel -> system properties -> environment variable -> path)

* Confirm whether the pip version meets the requirements. The pip version is required to be 9.0.1+

    python -m ensurepip

    python -m pip --version

* Confirm that Python and pip is 64 bit，and the processor architecture is x86_64(or x64、Intel 64、AMD64)architecture. Currently, PaddlePaddle doesn't support arm64 architecture. The first line of output from the following command should be "64bit", and the second line should be "x86_64", "x64" or "AMD64":

    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"


* The installation package provided by default requires the computer to support MKL. If your environment does not support MKL, please download the `openblas` version of the installation package in [here](./Tables.html#ciwhls-release)
* Nccl, distributed and other related functions are not supported in the current version.


## CPU or GPU

* If your computer doesn’t have NVIDIA® GPU, please install the CPU version of PaddlePaddle

* If your computer has NVIDIA® GPU, and it satisfies the following requirements, we recommend you to install the GPU version of PaddlePaddle
    * *CUDA Toolkit 9.0/10.0 with cuDNN v7.3+*
    * *GPU's computing capability exceeds 1.0*

Note: currently, the official Windows installation package only support CUDA 9.0/10.0 with single GPU, and don't include CUDA 9.1/9.2/10.1. if you need to use, please compile by yourself through the source code.

Please refer to the NVIDIA official documents for the installation process and the configuration methods of [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/).

## Installation Method

There are 2 ways to install PaddlePaddle on Windows:

* pip installation (recommended)
* [source code compilation and installation](./compile/compile_Windows.html/#win_source)

We would like to introduce the pip installation here.

## Installation steps

* CPU version of PaddlePaddle:
  * `python -m pip install paddlepaddle==2.0.0a0 -i https://mirror.baidu.com/pypi/simple`(Baidu source is recommended) or `python -m pip install paddlepaddle==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`

* GPU version of PaddlePaddle:
  * `python -m pip install paddlepaddle-gpu==2.0.0a0 -i https://mirror.baidu.com/pypi/simple`(Baidu source is recommended) or `python -m pip install paddlepaddle-gpu==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`

There is a checking function below for [verifyig whether the installation is successful](#check). If you have any further questions, please check the [FAQ](./FAQ.html).

Notice:

* `python -m pip install paddlepaddle-gpu==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple` This command will install PaddlePaddle that supports CUDA 10.0(with cuDNN v7.3+).
<a name="check"></a>
## Installation Verification
After completing the installation process, you can use `python` to enter python interface and input `import paddle.fluid as fluid` and then `fluid.install_check.run_check()` to check whether the installation is successful.

If you see `Your Paddle Fluid is installed succesfully!`, your installation is verified successful.

## Uninstall PaddlePaddle

* ***CPU version of PaddlePaddle***: `python -m pip uninstall paddlepaddle`

* ***GPU version of PaddlePaddle***: `python -m pip uninstall paddlepaddle-gpu`
