..  _install_introduction_:

=======================
 Installation Guide
=======================


----------------------
  Important updates
----------------------

* Add support for python3.9, and no longer supports python2.7 and python3.5
* Add support for CUDA 11.2, and no longer supports CUDA 9.0, CUDA 10.0 and CUDA 11.0
* Add support for ROCm platform (2.1 Paddle's support for ROCm platform is experimental)
* Linux system-related packages have been split into two types of packages, avx and noavx (Most machines use the avx instruction set. You can check whether your machine supports it through commands on the `PIP installation under Linux <pip/linux-pip.html>`_ page )
* Add a CPU image with jupyter pre-installed. Jupyter service will be started after starting the image
* Added support for Windows Visual Studio 2017 compilation, fully upgraded from VS2015 to VS2017
* Add support for NVIDIA NGC PaddlePaddle Container


------------------------
  Installation Manuals
------------------------


The manuals will guide you to build and install PaddlePaddle on your 64-bit desktop or laptop.

1. Operating system requirements:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

* Windows 7 / 8 / 10, Pro/Enterprise
* Ubuntu 16.04 / 18.04
* CentOS 7
* MacOS 10.11 / 10.12 / 10.13 / 10.14
* 64-bit operating system is required

2. Processor requirements:
>>>>>>>>>>>>>>>>>>>>>>>>>>

* Processor supports MKL
* The processor architecture is x86_64(or called x64, Intel 64, AMD64). Currently, PaddlePaddle does not support arm64.

3. Version requirements of python and pip:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

* Python requires version 3.7/3.8/3.9/3.10
* Python needs pip, and pip requires version 20.2.2 or above
* Python and pip requires 64-bit

4. PaddlePaddle's support for GPU:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

* Currently, **PaddlePaddle** supports **CUDA** driver of **NVIDIA** graphics card and **ROCm** driver of **AMD** card.
* You need to install `cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ , and version 7.6+ is required(For CUDA10.1/10.2)
* If you need GPU multi-card mode, you need to install `NCCL 2 <https://developer.nvidia.com/nccl/>`_

    * Only Ubuntu/CentOS support NCCL 2
* You need to install `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-windows/>`_ , depending on your system, there are different requirements for CUDA version:

    * Windows install GPU version

        * Windows 7 / 8 / 10 support CUDA 10.1/10.2/11.2 single-card mode, but don't support CUDA 9.1/9.2/10.1
        * don't support install using **nvidia-docker**
    * Ubuntu install GPU version

        * Ubuntu 16.04 supports CUDA 10.1/10.2/11.2
        * Ubuntu 18.04 supports CUDA 10.1/10.2/11.2
        * If you install using **nvidia-docker** , it supports CUDA 10.2/11.2
    * CentOS install GPU version

        * If you install using native **pip** :

            * CentOS 7 supports CUDA 10.1/10.2/11.2
        * If you compile and install using native source code:

            * CentOS 7 supports CUDA 10.1/10.2/11.2
        * If you install using  **nvidia-docker** , CentOS 7 supports CUDA 10.2/11.2
    * MacOS isn't supported: PaddlePaddle has no GPU support in Mac OS platform

Please make sure your environment meets the above conditions. If you have other requirements, please refer to `Appendix <Tables_en.html#ciwhls-release>`_ .

5. PaddlePaddle's support for NCCL:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

* Support for Windows

    * not support NCCL
* Support for Ubuntu

    * Ubuntu 16.04:

        * support NCCL v2.4.2-v2.4.8 under CUDA10.1
    * Ubuntu 18.04:

        * support v2.4.2-v2.4.8 under CUDA10.1
* Support for CentOS

    * CentOS 6: not support NCCL
    * CentOS 7:

        * support NCCL v2.4.2-v2.4.8 under CUDA10.1
* Support for MacOS

    * not support NCCL


The first way to install: use pip to install
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

You can choose any of the four ways to install: "use pip to install", "use Conda to install", "use Docker to install", "compiling from the source code"

This section describes how to use pip to install.

1. You need to confirm that your operating system meets the requirements listed above

2. You need to confirm that your processor meets the requirements listed above

3. Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

    Use the following command to output Python path. Depending on your environment, you may need to replace Python in all command lines in the description with specific Python path

        In the Windows environment, the command to output Python path is:

        ::

            where python

        In the MacOS/Linux environment, the command to output Python path is:

        ::

            which python


4. Check the version of Python

    Confirm the Python is 3.7/3.8/3.9/3.10 using command
    ::

        python --version

5. Check the version of pip and confirm it is 20.2.2 or above

    ::

        python -m ensurepip
        python -m pip --version


6. Confirm that Python and pip is 64 bit，and the processor architecture is x86_64(or called x64、Intel 64、AMD64)architecture. Currently, PaddlePaddle doesn't support arm64 architecture. The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64" :

    ::

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"


7. If you want to use `pip <https://pypi.org/project/pip/>`_ to install PaddlePaddle, you can use the command below directly:

    (1). **CPU version** : If you only want to install CPU version, please refer to command below

        Command to install CPU version is:
        ::

            python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

            or

            python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple


    (2). **GPU version** : If you only want to install GPU version, please refer to command below


        Note:

            * You need to confirm that your GPU meets the requirements listed above

        Please attention that PaddlePaddle installed through command below only supports CUDA10.2 under Windows、Ubuntu、CentOS:
        ::

            python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

            or

            python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple


    Please confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python. Depending on the environment, you may need to replace Python in all command lines in the instructions with Python 3 or specific Python path.

8. Verify installation

    After the installation is complete, you can use `python` to enter the Python interpreter and then use `import paddle` and then  `paddle.utils.run_check()` to verify that the installation was successful.

    If `PaddlePaddle is installed successfully!` appears, it means the installation was successful.


9. For more information to help, please refer to:

    `install under Ubuntu <pip/linux-pip_en.html>`_

    `install under MacOS <pip/macos-pip_en.html>`_

    `install under Windows <pip/windows-pip_en.html>`_



The second way to install: compile and install with container
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

- We recommend that you use `NVIDIA PaddlePaddle Container <https://www.paddlepaddle.org.cn/documentation/docs/zh/install/install_NGC_PaddlePaddle_ch.html>`_ for your development environment installation.
- Pros
    1. Lastest version of CUDA
    2. Newer verison of Ubuntu OS(18.04)
    3. Performance and development efficiency have been optimized by NVIDIA


The third way to install: compile and install with source code
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

- If you use PaddlePaddle only, we suggest you installation methods **pip** to install.
- If you need to develop PaddlePaddle, please refer to `compile from source code <compile/fromsource_en.html>`_

..  toctree::
    :hidden:

    pip/frompip_en.rst
    conda/fromconda_en.rst
    docker/fromdocker_en.rst
    compile/fromsource_en.rst
    install_Kunlun_en.md
    install_NGC_PaddlePaddle_en.rst
    Tables_en.md
