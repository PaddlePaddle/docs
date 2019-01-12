***

# **Install under Windows**

This instruction will show you how to install PaddlePaddle on a 64-bit desktop or laptop and Windows. The Windows systems we support must meet the following requirements.


Please note: Attempts on other systems may cause the installation to fail. Please ensure that your environment meets the conditions. The installation we provide by default requires your computer processor to support the AVX instruction set. Otherwise, please select the version of `no_avx` in [the multi-version whl package installation list](Tables.html/#ciwhls):

Windows can use software such as `cpu-z` to detect whether your processor supports the AVX instruction set.

The current version does not support NCCL, distributed, AVX, warpctc and MKL related functions.

* *Windows 7/8 and Windows 10 Professional/Enterprise Edition*

## Determine which version to install

* Under Windows, we currently only offer PaddlePaddle that supports CPU.

## Choose an installation method

### ***Install using pip***

We do not provide a quick installation command, please install according to the following steps： 

* First, **check that your computer and operating system** meet the following requirements:

		For python2: Python2.7.15 downloaded from official Python
		For python3: Use python3.5.x, python3.6.x or python3.7.x downloaded from official Python

* Python2.7.x :pip >= 9.0.1
* Python3.5.x, python3.6.x or python3.7.x :pip3 >= 9.0.1

Here's how to install PaddlePaddle:

* Use pip install to install PaddlePaddle:

    ** paddlepaddle's dependency package `recordio` may not be installed with `pip`'s default source, you can use `easy_install recordio` to install. **

	** For users who need **the CPU version PaddlePaddle**: `pip install paddlepaddle` or `pip3 install paddlepaddle`. **

Now you have completed the process of installing PaddlePaddle via `pip install`.

## ***Verify installation***

After completing the installation, you can use `python` or `python3` to enter the python interpreter and then use `import paddle.fluid` to verify that the installation was successful.

## ***How to uninstall***

Use the following command to uninstall PaddlePaddle (users who use Docker to install PaddlePaddle, please use the following command in the container containing PaddlePaddle):

* ***CPU version of PaddlePaddle***: `pip uninstallpaddlepaddle `or `pip3 uninstall paddlepaddle`
