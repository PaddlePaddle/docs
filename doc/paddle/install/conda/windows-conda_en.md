# Installation on Windows via Conda

[Anaconda](https://www.anaconda.com/)is a free and open source distribution of Python and R for computational science. Anaconda is dedicated to simplifying package management and deployment. Anaconda's packages are managed using the package management system Conda. Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux.



## Environmental preparation

Before performing PaddlePaddle installation, please make sure that your Anaconda software environment is properly installed. For software download and installation, see Anaconda's official website (https://www.anaconda.com/). If you have installed Anaconda correctly, follow these steps to install PaddlePaddle.

* Windows 7/8/10 Pro/Enterprise (64bit)
  * GPU Version supportCUDA 9.0/9.1/9.2/10.0/10.1/10.2/11.0，且仅支持单卡
* conda version 4.8.3+ (64 bit)



### 1.1 Create Virtual Environment

#### 1.1.1 Create the Anaconda Virtual Environment

Create virtual environment First create the Anaconda virtual environment according to the specific Python version. The Anaconda installation of PaddlePaddle supports the following four Python installation environments.

If you want to use python version 2.7:

```
conda create -n paddle_env python=2.7
```

If you want to use python version 3.5:

```
conda create -n paddle_env python=3.5
```

If you want to use python version 3.6:

```
conda create -n paddle_env python=3.6
```

If you want to use python version 3.7:

```
conda create -n paddle_env python=3.7
```

If you want to use python version 3.8:

```
conda create -n paddle_env python=3.8
```



#### 1.1.2 Enter the Anaconda Virtual Environment

for Windows

```
activate paddle_env
```

for MacOS/Linux

```
conda activate paddle_env
```



## 1.2 Confirm Other Environments

Confirm that your conda virtual environment and the Python loaction which is preapared to install PaddlePaddle are where you expected them for your computer may have multiple Pythons environments. Enter Anaconda's command line terminal and enter the following command to confirm the Python location.

1.2.1 If you are using Python 2, use the following command to get the Python path. Depending on your environment, you may need to replace python in all command lines in the instructions with specific Python path.

In a Windows environment, the command to get the Python path is:

```
where python
```

In a MacOS/Linux environment, the command to get the Python path is:

```
which python
```



If you are using Python 3, use the following command to get the Python path. Depending on your environment, you may need to replace python in all command lines in the instructions with specific Python path.

In a Windows environment, the command to get the Python path is:

```
where python3
```

In a MacOS/Linux environment, the command to get the Python path is:

```
which python3
```



1.2.2 Check the version of Python

If you are using Python 2, use the following command to confirm it's version is 2.7.15+

```
python --version
```

If you are using Python 3, use the following command to confirm it's version is 3.5.1+/3.6/3.7/3.8

```
python3 --version
```



1.2.3 Confirm that Python and pip are 64bit, and the processor architecture is x86_64 (or x64, Intel 64, AMD64) architecture. Currently PaddlePaddle does not support arm64 architecture. The first line below print "64bit", the second line prints "x86_64 (or x64, AMD64)."

If you are using Python2:

```
python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
```

If you are using Python3:

```
python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
```





## INSTALLATION

We will introduce conda installation here.

### Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install [the CPU Version of PaddlePaddle](#cpu)

* If your computer has NVIDIA® GPU, please make sure that the following conditions are met and install [the GPU Version of PaddlePaddle](#gpu)

  * **CUDA toolkit 9.0/10.0/10.1/10.2 with cuDNN v7.6+**

  * **CUDA toolkit 11.0 with cuDNN v8.0.4(**

  * **Hardware devices with GPU computing power over 1.0**

    You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)


### Installation Step

You can choose the following version of PaddlePaddle to start installation:

* [CPU Version of PaddlePaddle](#cpu)
* [GPU Version of PaddlePaddle](#gpu)
  * [CUDA9.0 PaddlePaddle](#cuda9)
  * [CUDA10.0 PaddlePaddle](#cuda10)
  * [CUDA10.1 PaddlePaddle](#cuda10.1)
  * [CUDA10.2 PaddlePaddle](#cuda10.2)
  * [CUDA11.0 PaddlePaddle](#cuda11)



#### 2.1 <span id="cpu">CPU version of PaddlePaddle</span>

```
conda install paddlepaddle==2.0.0rc1 -c paddle
```



#### 2.2<span id="gpu"> GPU version of PaddlePaddle</span>

*  <span id="cuda9">If you are using CUDA 9，cuDNN 7.6+:</span>

  ```
  conda install paddlepaddle-gpu==2.0.0rc1 cudatoolkit=9.0 -c paddle
  ```

* <span id="cuda10">If you are using CUDA 10.0，cuDNN 7.6+</span>

  ```
  conda install paddlepaddle-gpu==2.0.0rc1 cudatoolkit=10.0 -c paddle
  ```

*  <span id="cuda10.1">If you are using CUDA 10.1，cuDNN 7.6+</span>

  ```
  conda install paddlepaddle-gpu==2.0.0rc1 cudatoolkit=10.1 -c paddle
  ```

*  <span id="cuda10.2">If you are usingCUDA 10.2，cuDNN 7.6+:</span>

  ```
  conda install paddlepaddle-gpu==2.0.0rc1 cudatoolkit=10.2 -c paddle
  ```

*  <span id="cuda11">If you are using CUDA 11，cuDNN 8.0.4+:</span>

  ```
  conda install paddlepaddle-gpu==2.0.0rc1 cudatoolkit=11.0 -c paddle
  ```



## Verify installation

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.



## Notice

For domestic users who cannot connect to the Anaconda official source, you can add Tsinghua source to install it according to the following command.

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
conda config --set show_channel_urls yes
```
