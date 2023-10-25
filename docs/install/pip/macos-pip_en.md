# Install on MacOS via PIP

## Environmental preparation

### 1.1 How to check your environment

* You can use the following commands to view the local operating system and bit information

  ```
  uname -m && cat /etc/*release
  ```



* Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

  * Use the following command to output Python path. Depending on the environment, you may need to replace python3 in all command lines in the description with specific Python path

    ```
    which python3
    ```



* You need to confirm whether the version of Python meets the requirements

  * Use the following command to confirm that it is 3.7/3.8/3.9/3.10/3.11

        python3 --version

* It is required to confirm whether the version of pip meets the requirements. The version of pip is required to be 20.2.2 or above

    ```
    python3 -m ensurepip
    ```

    ```
    python3 -m pip --version
    ```


* You need to confirm that Python and pip are 64bit, and the processor architecture is x86_64(or called x64、Intel 64、AMD64) or arm64 (PaddlePaddle already supports Mac M1):


    ```
    python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```


* If you do not know the machine environment, please download and use[Quick install script](https://fast-install.bj.bcebos.com/fast_install.sh), for instructions please refer to[here](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## INSTALLATION

### Choose CPU/GPU

* Currently, only the CPU version of PaddlePaddle is supported in the MacOS environment


### Installation Step

You can choose the following version of PaddlePaddle to start installation:

* Please use the following command to install PaddlePaddle：


```
python3 -m pip install paddlepaddle==2.5.2 -i https://mirror.baidu.com/pypi/simple
```

Note：


* Please confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python. Depending on the environment, you may need to replace python3 in all command lines in the instructions with specific Python path.
* The above commands install the `avx` and `mkl` package by default. Paddle no longer supports `noavx` package. To determine whether your machine supports `avx`, you can use the following command. If the output contains `avx`, it means that the machine supports `avx`:
   ```
   sysctl machdep.cpu.features | grep -i avx
   ```
   or
   ```
   sysctl machdep.cpu.leaf7_features | grep -i avx
   ```



## Verify installation

After the installation is complete, you can use `python` to enter the Python interpreter and then use `import paddle` and `paddle.utils.run_check()`

If `PaddlePaddle is installed successfully!` appears, to verify that the installation was successful.

## How to uninstall

Please use the following command to uninstall PaddlePaddle:

```
python3 -m pip uninstall paddlepaddle
```
