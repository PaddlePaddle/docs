..  _install_introduction_:

============================
 Installation Manuals
============================

The manuals will guide you to build and install PaddlePaddle on your 64-bit desktop or laptop.

1. Operating system requirements:
=================================

* Windows 7 / 8 / 10, Pro/Enterprise
* Ubuntu 14.04 / 16.04 / 18.04
* CentOS 6 / 7
* MacOS 10.11 / 10.12 / 10.13 / 10.14
* 64-bit operating system is required

2. Processor requirements:
==========================

* Processor supports MKL
* The processor architecture is x86_64(or called x64, Intel 64, AMD64). Currently, PaddlePaddle does not support arm64.

3. Version requirements of python and pip:
==========================================

* Python 2 requires version 2.7.15+
* Python 3 requires version 3.5.1+/3.6/3.7
* Python needs pip, and pip requires version 9.0.1+
* Python and pip requires 64-bit

4. PaddlePaddle's support for GPU:
==================================

* Currently, **PaddlePaddle** only supports **CUDA** driver of **NVIDIA** graphics card.
* You need to install `cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ , and version 7.6+ is required(For CUDA9/10) 
* If you need GPU multi-card mode, you need to install `NCCL 2 <https://developer.nvidia.com/nccl/>`_

    * Only Ubuntu/CentOS support NCCL 2
* You need to install `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-windows/>`_ , depending on your system, there are different requirements for CUDA version:

    * Windows install GPU version

        * Windows 7 / 8 / 10 support CUDA 9.0 / 10.0 single-card mode, but don't support CUDA 9.1/9.2/10.1		
        * don't support install using **nvidia-docker** 
    * Ubuntu install GPU version

        * Ubuntu 14.04 supports CUDA 10.0/10.1, but doesn't support CUDA 9.0/9.1/9.2
        * Ubuntu 16.04 supports CUDA 9.0/9.1/9.2/10.0/10.1
        * Ubuntu 18.04 supports CUDA 10.0/10.1, but doesn't support CUDA 9.0/9.1/9.2
        * If you install using **nvidia-docker** , it supports CUDA 9.0/9.1/9.2/10.0/10.1
    * CentOS install GPU version

        * If you install using native **pip** :

            * CentOS 7 supports CUDA 9.0/9.1/9.2/10.0/10.1, CUDA 9.1 supports single-card mode only
            * CentOS 6 supports CUDA 9.0/9.1/9.2/10.0/10.1 single-card mode
        * If you compile and install using native source code:

            * CentOS 7 supports CUDA 9.0/9.1/9.2/10.0/10.1, CUDA 9.1 supports single-card mode only
            * CentOS 6 is not recommended, we don't provide official support in case of compilation problems
        * If you install using  **nvidia-docker** , CentOS 7 supports CUDA 9.0/9.1/9.2/10.0/10.1
    * MacOS isn't supported: PaddlePaddle has no GPU support in Mac OS platform

Please make sure your environment meets the above conditions. If you have other requirements, please refer to `Appendix <Tables_en.html#ciwhls>`_ .

5. PaddlePaddle's support for NCCL:
===================================

* Support for Windows

    * not support NCCL
* Support for Ubuntu

    * Ubuntu 14.04:

        * support NCCL v2.4.2-v2.4.8 under CUDA10.1 
        * support NCCL v2.3.7-v2.4.8 under CUDA10.0
    * Ubuntu 16.04:

        * support NCCL v2.4.2-v2.4.8 under CUDA10.1
        * support NCCL v2.3.7-v2.4.8 under CUDA10.0/9.2/9.0        
        * support NCCL v2.1.15 under CUDA9.1
    * Ubuntu 18.04:

        * support v2.4.2-v2.4.8 under CUDA10.1 
        * support NCCL v2.3.7-v2.4.8 under CUDA10.0 
* Support for CentOS

    * CentOS 6: not support NCCL
    * CentOS 7:

        * support NCCL v2.4.2-v2.4.8 under CUDA10.1 
        * support NCCL v2.3.7-v2.4.8 under CUDA10.0/9.2/9.0 
* Support for MacOS

    * not support NCCL


The first way to install: use pip to install
============================================

You can choose any of the four ways to install: "use pip to install", "use Conda to install", "use Docker to install", "compiling from the source code"

This section describes how to use pip to install.

1. You need to confirm that your operating system meets the requirements listed above

2. You need to confirm that your processor meets the requirements listed above

3. Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

    If you are using Python 2, use the following command to output Python path. Depending on your environment, you may need to replace Python in all command lines in the description with specific Python path
    
        In the Windows environment, the command to output Python path is:
        
        ::

            where python

        In the MacOS/Linux environment, the command to output Python path is:

        ::

            which python

    If you are using Python 3, use the following command to output Python path. Depending on your environment, you may need to replace Python in all command lines in the description with specific Python path

        In the Windows environment, the command to output Python path is:

        ::

            where python3

        In the MacOS/Linux environment, the command to output Python path is:

        ::

            which python3

4. Check the version of Python

    If you are using Python 2，confirm it is 2.7.15+ using command
    ::
    
        python --version

    If you are using Python 3，confirm it is 3.5.1+/3.6/3.7 using command
    ::
    
        python3 --version
    
5. Check the version of pip and confirm it is 9.0.1+  

    If you are using Python 2
    ::
    
        python -m ensurepip 
        python -m pip --version

    If you are using Python 3
    ::
    
        python3 -m ensurepip
        python3 -m pip --version

6. Confirm that Python and pip is 64 bit，and the processor architecture is x86_64(or called x64、Intel 64、AMD64)architecture. Currently, PaddlePaddle doesn't support arm64 architecture. The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64" :

    If you use Python 2
    ::

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

    If you use Python 3
    ::
    
        python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

7. If you want to use `pip <https://pypi.org/project/pip/>`_ to install PaddlePaddle, you can use the command below directly:

    (1). **CPU version** : If you only want to install CPU version, please refer to command below

        If you are using Python 2, command to install CPU version is:
        ::
    
            python -m pip install paddlepaddle==2.0.0b0 -i https://mirror.baidu.com/pypi/simple

            or

            python -m pip install paddlepaddle==2.0.0b0 -i https://pypi.tuna.tsinghua.edu.cn/simple

        If you are using Windows environment, please use the following instruction:

            python -m pip install paddlepaddle==2.0.0b0 -f https://paddlepaddle.org.cn/whl/stable.html 

        If you are using Python 3, please change **python** in the above command to **python3** and install.
        

    (2). **GPU version** : If you only want to install GPU version, please refer to command below


        Note:

            * You need to confirm that your GPU meets the requirements listed above

        If you are using Python2, please attention that PaddlePaddle installed through command below only supports CUDA10.0 under Windows、Ubuntu、CentOS:
        ::

            python -m pip install paddlepaddle-gpu==2.0.0b0 -i https://mirror.baidu.com/pypi/simple

            or

            python -m pip install paddlepaddle-gpu==2.0.0b0 -i https://pypi.tuna.tsinghua.edu.cn/simple

        If you are using Windows environment, please use the following instruction:

            python -m pip install paddlepaddle_gpu==2.0.0b0 -f https://paddlepaddle.org.cn/whl/stable.html
        
        If you are using Python 3, please change **python** in the above command to **python3** and install.

8. Verify installation

    After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid as fluid` and then  `fluid.install_check.run_check()` to verify that the installation was successful.

    If `Your Paddle Fluid is installed succesfully!` appears, it means the installation was successful.


9. For more information to help, please refer to:

    `install under Ubuntu <install_Ubuntu_en.html>`_

    `install under CentOS <install_CentOS_en.html>`_

    `install under MacOS <install_MacOS_en.html>`_

    `install under Windows <install_Windows_en.html>`_


The second way to install: compile and install with source code
===============================================================

- If you use PaddlePaddle only, we suggest you installation methods **pip** to install.
- If you need to develop PaddlePaddle, please refer to `compile from source code <compile/fromsource.html>`_

..	toctree::
	:hidden:

	install_Ubuntu_en.md
	install_CentOS_en.md
	install_MacOS_en.md
	install_Windows_en.md
	compile/fromsource_en.rst
	Tables_en.md
