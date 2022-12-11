# Install on macOS via PIP

## Environmental preparation

### 1.1 PREQUISITES

* **macOS version 10.11/10.12/10.13/10.14 (64 bit) (not support GPU version)**

* **Python version 3.7/3.8/3.9/3.10 (64 bit)**

* **pip or pip3 版本 20.2.2 or above (64 bit)**


### 1.2 How to check your environment

* You can use the following commands to view the local operating system and bit information

  ```
  uname -m && cat /etc/*release
  ```



* Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

  * Use the following command to output Python path. Depending on the environment, you may need to replace Python in all command lines in the description with specific Python path

    ```
    which python
    ```



* You need to confirm whether the version of Python meets the requirements

  * Use the following command to confirm that it is 3.7/3.8/3.9/3.10

        python --version

* It is required to confirm whether the version of pip meets the requirements. The version of pip is required to be 20.2.2 or above

    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```


* You need to confirm that Python and pip are 64bit, and the processor architecture is x86_64(or called x64、Intel 64、AMD64). Currently, paddlepaddle does not support arm64 architecture. The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64"


    ```
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```


* The installation package provided by default requires computer support for MKL

* If you do not know the machine environment, please download and use[Quick install script](https://fast-install.bj.bcebos.com/fast_install.sh), for instructions please refer to[here](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/install_script.md)。



## INSTALLATION

If you installed Python via Homebrew or the Python website, `pip` was installed with it. If you installed Python 3.x, then you will be using the command `pip3`. We will introduce pip installation here.

### Choose CPU/GPU

* Currently, only the CPU version of PaddlePaddle is supported in the macOS environment


### Installation Step

You can choose the following version of PaddlePaddle to start installation:

* Please use the following command to install PaddlePaddle：


```
python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html
```

Note：


* Please confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python. Depending on the environment, you may need to replace Python in all command lines in the instructions with specific Python path.



## Verify installation

After the installation is complete, you can use `python` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

```
python -m pip uninstall paddlepaddle
```
