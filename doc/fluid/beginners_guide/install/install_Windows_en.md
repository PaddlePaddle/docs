***

# **Install on Windows**

This instruction will show you how to install PaddlePaddle on Windows.  The following conditions must be met before you begin to install:

* *a 64-bit desktop or laptop*
* *Windows 7/8 , Windows 10 Professional/Enterprise Edition*

**Note** : 

* The current version does not support NCCL, distributed training, AVX, warpctc and MKL related functions.

* Currently, only PaddlePaddle for CPU is supported on Windows.




## Installation Steps  

### ***Install through pip***

* Check your Python versions

Python2.7.15，Python3.5.x，Python3.6.x，Python3.7.x on [Official Python](https://www.python.org/downloads/) are supported.
 
* Check your pip version

Version of pip or pip3 should be equal to or above 9.0.1 .

* Install PaddlePaddle

Execute `pip install paddlepaddle` or `pip3 install paddlepaddle` to download and install PaddlePaddle.


## ***Verify installation***

After completing the installation, you can use `python` or `python3` to enter the python interpreter and then use `import paddle.fluid` to verify that the installation was successful.

## ***How to uninstall***

Use the following command to uninstall PaddlePaddle : `pip uninstallpaddlepaddle `or `pip3 uninstall paddlepaddle`

