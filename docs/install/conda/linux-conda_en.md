# Installation on Linux via Conda

[Anaconda](https://www.anaconda.com/)is a free and open source distribution of Python and R for computational science. Anaconda is dedicated to simplifying package management and deployment. Anaconda's packages are managed using the package management system Conda. Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux.


## Environmental preparation

### 1.1 Create Virtual Environment

#### 1.1.1 Create the Anaconda Virtual Environment

Create virtual environment First create the Anaconda virtual environment according to the specific Python version. The Anaconda installation of PaddlePaddle supports Python version of 3.6 - 3.10.

```
conda create -n paddle_env python=YOUR_PY_VER
```



#### 1.1.2 Enter the Anaconda Virtual Environment

```
conda activate paddle_env
```



### 1.2 Confirm Other Environments

Confirm that your conda virtual environment and the Python loaction which is preapared to install PaddlePaddle are where you expected them for your computer may have multiple Pythons environments. Enter Anaconda's command line terminal and enter the following command to confirm the Python location.

#### 1.2.1 Confirm the installation path of python

Depending on your environment, you may need to replace python3 in all command lines in the instructions with specific Python path.

The command to get the Python path is:

```
which python3
```



#### 1.2.2 Check the version of Python


Use the following command to confirm it's version

```
python3 --version
```



#### 1.2.3 Check the system environment

Confirm that Python and pip are 64bit, and the processor architecture is x86_64 (or x64, Intel 64, AMD64) architecture. The first line below print "64bit", the second line prints "x86_64 (or x64, AMD64)."


```
python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
```





## INSTALLATION

### Add Tsinghua source (optional)

For domestic users who cannot connect to the Anaconda official source, you can add Tsinghua source according to the following command.


```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```
```
conda config --set show_channel_urls yes
```


### Installation Step

You can choose the following version of PaddlePaddle to start installation:



#### CPU Version of PaddlePaddle

If your computer doesn't have NVIDIA® GPU, please install `the CPU Version of PaddlePaddle`

```
conda install paddlepaddle==2.4.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```



#### GPU Version of PaddlePaddle


*  If you are usingCUDA 10.2，cuDNN 7.6.5(for multi card support, NCCL>=2.7):

  ```
  conda install paddlepaddle-gpu==2.4.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
  ```

*  If you are using CUDA 11.2，cuDNN 8.2.1(for multi card support, NCCL>=2.7):

  ```
  conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
  ```

*  If you are using CUDA 11.6，cuDNN 8.4.0(for multi card support, NCCL>=2.7):

  ```
  conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
  ```

*  If you are using CUDA 11.7，cuDNN 8.4.1(for multi card support, NCCL>=2.7):

  ```
  conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
  ```

You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)


## Verify installation

After the installation is complete, you can use `python3` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.
