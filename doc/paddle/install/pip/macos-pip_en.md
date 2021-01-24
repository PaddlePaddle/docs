# Install on MacOS via PIP

## Environmental preparation

### 1.1 PREQUISITES

* **MacOS version 10.11/10.12/10.13/10.14 (64 bit) (not support GPU version)**

* **Python version 2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**

* **pip or pip3 版本 20.2.2+ (64 bit)**


### 1.2 How to check your environment

* You can use the following commands to view the local operating system and bit information

  ```
  uname -m && cat /ect/*release
  ```



* Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

  * If you are using Python 2, use the following command to output Python path. Depending on the environment, you may need to replace Python in all command lines in the description with specific Python path

    ```
    which python
    ```

  * If you are using Python 3, use the following command to output Python path. Depending on your environment, you may need to replace Python 3 in all command lines in the instructions with Python or specific Python path

    ```
    which python3
    ```



* You need to confirm whether the version of Python meets the requirements

  * If you are using Python 2, use the following command to confirm that it is 2.7.15+

        python --version

  * If you are using Python 3, use the following command to confirm that it is 3.5.1+/3.6/3.7/3.8

        python3 --version

* It is required to confirm whether the version of pip meets the requirements. The version of pip is required to be 20.2.2+

  * If you are using Python 2

    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```

  * If you are using Python 3

    ```
    python3 -m ensurepip
    ```

    ```
    python3 -m pip --version
    ```



* You need to confirm that Python and pip are 64bit, and the processor architecture is x86_64(or called x64、Intel 64、AMD64). Currently, paddlepaddle does not support arm64 architecture. The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64"

  * If you are using Python 2

    ```
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```

  * If you are using Python 3

    ```
    python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```



* The installation package provided by default requires computer support for MKL

* If you do not know the machine environment, please download and use[Quick install script](https://fast-install.bj.bcebos.com/fast_install.sh), for instructions please refer to[here](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## INSTALLATION

If you installed Python via Homebrew or the Python website, `pip` was installed with it. If you installed Python 3.x, then you will be using the command `pip3`. We will introduce pip installation here.

### Choose CPU/GPU

* Currently, only the CPU version of PaddlePaddle is supported in the MacOS environment


### Installation Step

You can choose the following version of PaddlePaddle to start installation:

* Please use the following command to install PaddlePaddle：

* If you are using Python 2:

```
python -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple
```

* If you are using Python 3:

```
python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple
```



## Verify installation

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

```
python -m pip uninstall paddlepaddle
```

```
python3 -m pip uninstall paddlepaddle
```
