***

# **Conda Installation**

Anaconda is a free and open source distribution of Python and R for computational science. Anaconda is dedicated to simplifying package management and deployment. Anaconda's packages are managed using the package management system Conda. Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux.

## Environmental preparation

Before performing PaddlePaddle installation, please make sure that your Anaconda software environment is properly installed. For software download and installation, see Anaconda's official website (https://www.anaconda.com/). If you have installed Anaconda correctly, follow these steps to install PaddlePaddle.

## Installation steps

1.Create virtual environment
First create the Anaconda virtual environment according to the specific Python version. The Anaconda installation of PaddlePaddle supports the following four Python installation environments.

If you want to use python version 2.7: `conda create -n paddle_env python=2.7`

If you want to use python version 3.5: `conda create -n paddle_env python=3.5`

If you want to use python version 3.6: `conda create -n paddle_env python=3.6`

If you want to use python version 3.7: `conda create -n paddle_env python=3.7`

If you want to use python version 3.8: `conda create -n paddle_env python=3.8`

Activate paddle_env (for Windows) or conda activate paddle_env (for MacOS / Linux) command to enter the Anaconda virtual environment.

2.Confirm that your conda virtual environment and the Python loaction which is preapared to install PaddlePaddle are where you expected them for your computer may have multiple Pythons environments. Enter Anaconda's command line terminal and enter the following command to confirm the Python location.

If you are using Python 2, use the following command to get the Python path. Depending on your environment, you may need to replace python in all command lines in the instructions with specific Python path.

     In a Windows environment, the command to get the Python path is:  where python

     In a MacOS/Linux environment, the command to get the Python path is:  which python

If you are using Python 3, use the following command to get the Python path. Depending on your environment, you may need to replace python in all command lines in the instructions with specific Python path.

     In a Windows environment, the command to get the Python path is:  where python3

     In a MacOS/Linux environment, the command to get the Python path is:  which python3


3.Check the version of Python

If you are using Python 2, use the following command to confirm it's version is  2.7.15+

     python --version

If you are using Python 3, use the following command to confirm it's version is  3.5.1+/3.6/3.7/3.8

     python3 --version

Confirm that Python and pip are 64bit, and the processor architecture is x86_64 (or x64, Intel 64, AMD64) architecture. Currently PaddlePaddle does not support arm64 architecture. The first line below print "64bit", the second line prints "x86_64 (or x64, AMD64)."

If you are using Python2:

      python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

If you are using Python3:

      python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

5.Install PaddlePaddle

 > (1) CPU version: If you just want to install the CPU version, please refer to the following command to install:

      conda install paddlepaddle

 > (2) GPU version: If you want to use the GPU version, please refer to the following command to install:

      If you are using CUDA 9, cuDNN 7.6+, the command to install the GPU version is:
           conda install paddlepaddle-gpu cudatoolkit=9.0

      If you are using CUDA 10.0, cuDNN 7.6+, the command to install the GPU version is:
           conda install paddlepaddle-gpu cudatoolkit=10.0

      If you are using CUDA 10.0, cuDNN 7.6+, the command to install the GPU version is:
           conda install paddlepaddle-gpu cudatoolkit=10.1

      If you are using CUDA 10.0, cuDNN 7.6+, the command to install the GPU version is:
           conda install paddlepaddle-gpu cudatoolkit=10.2

6.Installation environment verification

Use python to enter the python interpreter, enter import paddle.fluid, and then enter paddle.fluid.install_check.run_check (). If "Your Paddle Fluid is installed succesfully!" Appears, you have successfully installed.

## Notice

For domestic users who cannot connect to the Anaconda official source, you can add Tsinghua source to install it according to the following command.


     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
     conda config --set show_channel_urls yes
