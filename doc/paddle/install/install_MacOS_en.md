***
# **Install on MacOS**

## Environment preparation

* **MacOS version 10.11/10.12/10.13/10.14 (64 bit)(not support GPU version)**
* **Python version 2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**
* **pip or pip3 version 20.2.2+ (64 bit)**

### Note

* Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

    * If you are using Python 2, use the following command to output Python path. Depending on the environment, you may need to replace `python` in all command lines

        which python

    * If you are using Python 3, use the following command to output Python path. Depending on the environment, you may need to replace `python` in all command lines

        which python3

* You need to confirm whether the version of Python meets the requirements

    * If you are using Python 2, use the following command to confirm that it is 2.7.15+

        python --version

    * If you are using Python 3, use the following command to confirm that it is 3.5.1+/3.6/3.7/3.8

        python3 --version

* It is required to confirm whether the pip version meets the requirements. The pip version is required to be 20.2.2+

    * If you are using Python 2

        python -m ensurepip

        python -m pip --version

    * If you are using Python 3

        python3 -m ensurepip

        python3 -m pip --version

* Confirm that Python and pip is 64 bit，and the processor architecture is x86_64(or x64、Intel 64、AMD64)architecture. Currently, PaddlePaddle doesn't support arm64 architecture. The first line of output from the following command should be "64bit", and the second line should be "x86_64", "x64" or "AMD64".

    * If you are using Python 2

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

    * If you are using Python 3

        python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

* The installation package provided by default requires computer support for MKL

## Choose CPU/GPU

* Currently, only the CPU version of PaddlePaddle is supported in the MacOS environment

## Choose an installation method

Under the MacOS system we offer 3 installation methods:

* Pip installation (recommend)
* [Source code compilation and installation](./compile/compile_MacOS.html#mac_source)
* [Docker source code compilation and installation](./compile/compile_MacOS.html#mac_docker)


We will introduce pip installation here.

## Installation steps

* CPU version of PaddlePaddle：
  * For Python 2: `python -m pip install paddlepaddle==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple` or `python -m pip install paddlepaddle==2.0.0rc0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
  * For Python 3: `python3 -m pip install paddlepaddle==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple` or `python3 -m pip install paddlepaddle==2.0.0rc0 -i https://pypi.tuna.tsinghua.edu.cn/simple`

You can[Verify installation succeeded or not](#check), if you have any questions, please check[FAQ](./FAQ.html)

Note:

* On MacOS you need to install unrar to support PaddlePaddle, you can use command `brew install unrar`
* For python2.7, we suggest command `python`; for python3.x, we suggest command `python3`
* Download the latest release installation package by default. To obtain the development installation package, please refer to [here](./Tables.html#ciwhls)
* Using Python native to MacOS can cause installation failures. For **Python2**，we recommend to use [Homebrew](https://brew.sh) or python2.7.15 provided by [Python.org](https://www.python.org/ftp/python/2.7.15/python-2.7.15-macosx10.9.pkg); for **Python3**, please use python3.5.x、python3.6.x or python3.7.x provided by [Python.org](https://www.python.org/downloads/mac-osx/).

<a name="check"></a>
<br/><br/>
## ***Verify installation***

After the installation is completed, you can use `python` or `python3` to enter the Python interface, input `import paddle.fluid as fluid` and then `fluid.install_check.run_check()` to verify that the installation was successful.

If `Your Paddle Fluid is installed succesfully!` appears, it means the installation was successful.

<br/><br/>
## ***How to uninstall***

Please use the following command to uninstall PaddlePaddle：

* `python -m pip uninstall paddlepaddle` or `python3 -m pip uninstall paddlepaddle`
