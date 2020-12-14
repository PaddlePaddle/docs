..  _install_introduction_:
======================
 Installation Manuals
======================

The manuals will guide you to build and install PaddlePaddle on your 64-bit desktop or laptop.

1. Operating system requirements:
============================

* Windows 7 / 8 / 10, Pro/Enterprise
* Ubuntu 14.04 / 16.04 / 18.04
* CentOS 6 / 7
* MacOS 10.11 / 10.12 / 10.13 / 10.14
* 64-bit operating system is required

2. Processor requirements:
============================

* Processor supports MKL
* The processor architecture is x86_64(or called x64, Intel 64, AMD64). Currently, PaddlePaddle does not support arm64.

3. Version requirements of python and pip:
============================

* Python 2 requires version 2.7.15+
* Python 3 requires version 3.5.1+/3.6/3.7/3.8
* Python needs pip, and pip requires version 20.2.2+
* Python and pip requires 64-bit

4. PaddlePaddle's support for GPU:
=================================

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
=================================

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
================================

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

    If you are using Python 3，confirm it is 3.5.1+/3.6/3.7/3.8 using command
    ::
    
        python3 --version
    
5. Check the version of pip and confirm it is 20.2.2+  

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
    
            python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

            or

            python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
        
        If you are using Python 3, command to install CPU version is:
        ::
    
            python3 -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

            or

            python3 -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

    (2). **GPU version** : If you only want to install GPU version, please refer to command below


        Note:

            * You need to confirm that your GPU meets the requirements listed above

        If you are using Python2, please attention that PaddlePaddle installed through command below supports CUDA10.0 under Windows、Ubuntu、CentOS by default:
        ::

            python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

            or

            python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

        If you are using Python 2, CUDA 9, cuDNN 7.3+, command to install GPU version:
        ::

            python -m pip install paddlepaddle-gpu==1.8.5.post97 -i https://mirror.baidu.com/pypi/simple

            or

            python -m pip install paddlepaddle-gpu==1.8.5.post97 -i https://pypi.tuna.tsinghua.edu.cn/simple

        If you are using Python 2, CUDA 10.0, cuDNN 7.3+, command to install GPU version:
        ::

            python -m pip install paddlepaddle-gpu==1.8.5.post107 -i https://mirror.baidu.com/pypi/simple

            or

            python -m pip install paddlepaddle-gpu==1.8.5.post107 -i https://pypi.tuna.tsinghua.edu.cn/simple
        
        If you are using Python 3, please change **python** in the above command to **python3** and install.

8. Verify installation

    After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid as fluid` and then  `fluid.install_check.run_check()` to verify that the installation was successful.

    If `Your Paddle Fluid is installed succesfully!` appears, it means the installation was successful.


9. For more information to help, please refer to:

    `Install on Linux via pip <frompip_en/linux-pip_en.html>`_

    `Install on MacOS via pip <frompip_en/macos-pip_en.html>`_

    `Install on Windows via pip <frompip_en/windows-pip_en.html>`_


The second way to install: use Conda to install
================================

You can choose any of the four ways to install: "use pip to install", "use Conda to install", "use Docker to install", "compiling from the source code"

This section describes how to use Conda to install.

1. You need to confirm that your operating system meets the requirements listed above

2. You need to confirm that your processor meets the requirements listed above

3. Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python


3. For domestic users unable to connect to the official source of anaconda, you can add Tsinghua source for installation according to the following command.

    ::

        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
        conda config --set show_channel_urls yes

4. If you need to create a new Conda virtual environment specifically for Paddle to use (the environment name after --name, you can choose by yourself):

    If you are using Python2 under Window
    
    ::

        conda create --name paddle python=2.7
        activate paddle

    If you are using Python2 under MacOS/Linux

    ::

        conda create --name paddle python=2.7
        conda activate paddle

   If you are using Python3 under Window, note: python3 version can be 3.5.1+/3.6/3.7/3.8

    ::

        conda create --name paddle python=3.7
        activate paddle

    If you are using Python3 under MacOS/Linux, note: python3 version can be 3.5.1+/3.6/3.7/3.8

    ::

        conda create --name paddle python=3.7
        conda activate paddle

5. Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python, enter Anaconda's command-line terminal, enter the following instructions to confirm the Python location

    If you use Python 2, use the following command to output Python path. Depending on your environment, you may need to replace Python in all command lines in the description with specific Python path
        
        In the Windows environment, the command to output Python path is:
        
        ::

            where python

        In the MacOS/Linux environment, the command to output Python path is:

        ::

            which python

    If you use Python 3, use the following command to output Python path. Depending on your environment, you may need to replace Python3 in all command lines in the description with Python or specific Python path

        In the Windows environment, the command to output Python path is:
        
        ::

            where python3

        In the MacOS/Linux environment, the command to output Python path is:

        ::

            which python3

6. Check the version of Python

    If you are using Python 2, use the following command to confirm it is 2.7.15+
    ::
    
        python --version

    If you are using Python 3, use the following command to confirm it is 3.5.1+/3.6/3.7/3.8
    ::
    
        python3 --version
    
7. Check the version of pip and confirm it is 20.2.2+  

    If you are using Python 2
    ::
    
        python -m ensurepip 
        python -m pip --version

    If you are using Python 3
    ::
    
        python3 -m ensurepip
        python3 -m pip --version

8. Confirm Python and pip is 64 bit, and the processor architecture is x86_64(or called x64,Intel 64,AMD64) architecture. Currently, PaddlePaddle doesn't support arm64 architecture. The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64":

    If you are using Python 2
    ::

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

    If you are using Python 3
    ::
    
        python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

9. If you want to use Conda to install PaddlePaddle, you can directly use commands below:

    (1). **CPU version** :If you just want to install the CPU version, please refer to the following command installation

    ::

        conda install paddlepaddle
    

    (2). **GPU version** :If you just want to install the GPU version, please refer to the following command installation

        Note:

            * You need to confirm that your GPU meets the requirements listed above

        If you are using CUDA 8，cuDNN 7.1+, the command to install GPU version:
        ::
    
            conda install paddlepaddle-gpu cudatoolkit=8.0

        If you are using CUDA 9，cuDNN 7.3+, the command to install GPU version:
        ::
    
            conda install paddlepaddle-gpu cudatoolkit=9.0
        

        If you are using CUDA 10.0，cuDNN 7.3+, the command to install GPU version:：
        ::
    
            conda install paddlepaddle-gpu cudatoolkit=10.0

10. Verify installation

    After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid as fluid` and then  `fluid.install_check.run_check()` to verify that the installation was successful.

    If `Your Paddle Fluid is installed succesfully!` appears, it means the installation was successful.


11. For more information to help, please refer to:

    `Install on Linux via conda <fromconda_en/linux-conda_en.html>`_

    `Install on MacOS via conda <fromconda_en/macos-conda_en.html>`_

    `Install on Windows via <fromconda_en/windows-conda_en.html>`_


The third way to install: use Docker to install
================================

You can choose any of the four ways to install: "use pip to install", "use Conda to install", "use Docker to install", "compiling from the source code"

This section describes how to use Docker to install.

If you want to use `docker <https://www.docker.com>`_ to install PaddlePaddle, you can use command below:

1. **CPU version**

    (1). At first you need to install `docker <https://www.docker.com>`_

    Note:

        * CentOS 6 not support docker installation

        * processor need supporting MKL

    (2). Pull the image of the preinstalled PaddlePaddle:
    ::

        docker pull hub.baidubce.com/paddlepaddle/paddle:1.8.5

    (3). Use the image to build and enter the Docker container:
    ::

        docker run --name paddle -it -v dir1:dir2 hub.baidubce.com/paddlepaddle/paddle:1.8.5 /bin/bash

        > --name [Name of container] set the name of Docker;

        > -it The parameter indicates that the container has been operated interactively with the local machine;

        > -v Parameter is used to share files between the host and the container. dir1 is the host directory and dir2 is the directory mounted inside the container. Users can customize their own mounting directory by setting dir1 and dir2.For example, $PWD:/paddle specifies to mount the current path of the host (PWD variable in Linux will expand to the absolute path of the current path) to the /paddle directory inside the container; 

        > hub.baidubce.com/paddlepaddle/paddle:1.8.5 is the image name you need to use；/bin/bash is the command to be executed in Docker

2. **GPU version**

    (1). At first you need to install `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_

    Note:

        * processor need supporting MKL

        * Your computer needs to have NVIDIA graphics card supporting CUDA driver

        * You need to install `cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ ，version requires 7.3+(For CUDA9/10), 7.1+(For CUDA 8)

        * If you need GPU multi-card mode, you need to install `NCCL 2 <https://developer.nvidia.com/nccl/>`_

            * Only Ubuntu/CentOS support NCCL 2 technology

        * You need to install `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-windows/>`_ , depending on your system, there are different requirements for CUDA version:

            * Ubuntu/CentOS 7 ，if you use nvidia-docker to install, CUDA 8.0/9.0/9.1/9.2/10.0 is supported

            * Windows/MacOS/CentOS 6 not support nvidia-docker to install


    (2). Pull the image that supports CUDA 10.0, cuDNN 7.3 + pre installed PaddlePaddle:
    ::

        nvidia-docker pull hub.baidubce.com/paddlepaddle/paddle:1.8.5-gpu-cuda10.0-cudnn7

    (3). Use the image to build and enter the docker container:
    ::

        nvidia-docker run --name paddle -it -v dir1:dir2 hub.baidubce.com/paddlepaddle/paddle:1.8.5-gpu-cuda10.0-cudnn7 /bin/bash

        > --name [Name of container] set name of Docker;

        > -it The parameter indicates that the container has been operated interactively with the local machine;

        > -v Parameter is used to share files between the host and the container. dir1 is the host directory and dir2 is the directory mounted inside the container. Users can customize their own mounting directory by setting dir1 and dir2.For example, $PWD:/paddle specifies to mount the current path of the host (PWD variable in Linux will expand to the absolute path of the current path) to the /paddle directory inside the container;

        > hub.baidubce.com/paddlepaddle/paddle:1.8.5 is the image name you need to use；/bin/bash is the command to be executed in Docker

    Or if you need the version supporting **CUDA 9**, replace **cuda10.0** of the above command with **cuda9.0** 

3. If your machine is not in China's mainland                                                                                                              , you can pull the image directly from DockerHub:

    ::

        docker run --name paddle -it -v dir1:dir2 paddlepaddle/paddle:1.8.5 /bin/bash

        > --name [Name of container] set name of Docker;

        > -it The parameter indicates that the container has been operated interactively with the local machine;

        > -v Parameter is used to share files between the host and the container. dir1 is the host directory and dir2 is the directory mounted inside the container. Users can customize their own mounting directory by setting dir1 and dir2.For example, $PWD:/paddle specifies to mount the current path of the host (PWD variable in Linux will expand to the absolute path of the current path) to the /paddle directory inside the container; 

        > paddlepaddle/paddle:1.8.5 is the image name you need to use；/bin/bash is the command to be executed in docker

4. Verify installation

    After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid as fluid` and then  `fluid.install_check.run_check()` to verify that the installation was successful.

    If `Your Paddle Fluid is installed succesfully!` appears, it means the installation was successful.


5. For more help, refer to:

   `use Docker to install <fromdocker_en/docker_en.html>`_

	
The fourth way to install: compile and install with source code
====================================

- If you use PaddlePaddle only, we suggest you to choose one of the three installation methods **pip**, **conda**, **docker** to install.
- If you need to develop PaddlePaddle, please refer to `compile from source code <compile/fromsource.html>`_

..	toctree::
	:hidden:

	pip/frompip_en.rst
	conda/fromconda_en.rst
	compile/fromsource_en.rst
	docker/fromdocker_en.rst
	Tables_en.md
