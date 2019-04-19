***

# **Install on Windows**

This instruction will show you how to install PaddlePaddle on Windows.  The following conditions must be met before you begin to install:

* *a 64-bit desktop or laptop*
* *Windows 7/8 , Windows 10 Professional/Enterprise Edition*

**Note** : 

* The current version does not support NCCL, distributed training related functions.





## Installation Steps  

### ***Install through pip***

* Check your Python versions

Python2.7.15，Python3.5.x，Python3.6.x，Python3.7.x on [Official Python](https://www.python.org/downloads/) are supported.
 
* Check your pip version

Version of pip or pip3 should be equal to or above 9.0.1 .

* Install PaddlePaddle

* ***CPU version of PaddlePaddle***:
Execute `pip install paddlepaddle` or `pip3 install paddlepaddle` to download and install PaddlePaddle.

* ***GPU version of PaddlePaddle***:
Execute `pip install paddlepaddle-gpu`(python2.7) or `pip3 install paddlepaddle-gpu`(python3.x) to download and install PaddlePaddle.
 
## ***Verify installation***

After completing the installation, you can use `python` or `python3` to enter the python interpreter and then use `import paddle.fluid` to verify that the installation was successful.

## ***How to uninstall***

* ***CPU version of PaddlePaddle***:
Use the following command to uninstall PaddlePaddle : `pip uninstallpaddlepaddle `or `pip3 uninstall paddlepaddle`

* ***GPU version of PaddlePaddle***:
Use the following command to uninstall PaddlePaddle : `pip uninstall paddlepaddle-gpu` or `pip3 uninstall paddlepaddle-gpu`
