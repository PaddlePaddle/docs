# Installation on macOS via Conda

[Anaconda](https://www.anaconda.com/)is a free and open source distribution of Python and R for computational science. Anaconda is dedicated to simplifying package management and deployment. Anaconda's packages are managed using the package management system Conda. Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux.



## Environmental preparation

Before performing PaddlePaddle installation, please make sure that your Anaconda software environment is properly installed. For software download and installation, see Anaconda's official website (https://www.anaconda.com/). If you have installed Anaconda correctly, follow these steps to install PaddlePaddle.

* macOS version 10.11/10.12/10.13/10.14 (64 bit)(not support GPU version)
* conda version 4.8.3+ (64 bit)



### 1.1 Create Virtual Environment

#### 1.1.1 Create the Anaconda Virtual Environment

Create virtual environment First create the Anaconda virtual environment according to the specific Python version. The Anaconda installation of PaddlePaddle supports the following four Python installation environments.


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

If you want to use python version 3.9:

```
conda create -n paddle_env python=3.9
```



#### 1.1.2 Enter the Anaconda Virtual Environment

for Windows

```
activate paddle_env
```

for macOS/Linux

```
conda activate paddle_env
```



## 1.2 Confirm Other Environments

Confirm that your conda virtual environment and the Python loaction which is preapared to install PaddlePaddle are where you expected them for your computer may have multiple Pythons environments. Enter Anaconda's command line terminal and enter the following command to confirm the Python location.

1.2.1 Depending on your environment, you may need to replace python in all command lines in the instructions with specific Python path.

In a Windows environment, the command to get the Python path is:

```
where python
```

In a macOS/Linux environment, the command to get the Python path is:

```
which python
```


1.2.2 Check the version of Python

Use the following command to confirm it's version is 3.7/3.8/3.9/3.10

```
python --version
```



1.2.3 Confirm that Python and pip are 64bit, and the processor architecture is x86_64 (or x64, Intel 64, AMD64) architecture. Currently PaddlePaddle does not support arm64 architecture. The first line below print "64bit", the second line prints "x86_64 (or x64, AMD64)."


```
python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
```



## INSTALLATION

We will introduce conda installation here.

### Choose CPU/GPU

* Currently, only the CPU version of PaddlePaddle is supported in the macOS environment

### Installation Step

You can choose the following version of PaddlePaddle to start installation:

* Please use the following command to install PaddlePaddle：

  ```
  conda install paddlepaddle --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
  ```


## Verify installation

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.




## Notice

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
