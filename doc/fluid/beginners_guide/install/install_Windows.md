***

# **Windows下安装**

本文将介绍如何在Windows系统下安装PaddlePaddle，您的计算机需要满足以下要求：

* *64位台式机或笔记本电脑*

* *Windows 7/8 ，Windows 10 专业版/企业版*

注：

* 当前版本暂不支持NCCL，分布式，AVX，warpctc和MKL相关功能

* Windows环境下，目前仅支持CPU版本的PaddlePaddle

## 安装步骤

### ***使用pip安装***

* 对Python版本的要求

我们支持[Python官方提供](https://www.python.org/downloads/)的Python2.7.15，Python3.5.x，Python3.6.x，Python3.7.x

* 对pip版本的要求

您的pip或pip3版本号需不低于9.0.1

* 开始安装

执行如下命令：`pip install paddlepaddle` 或 `pip3 install paddlepaddle` 安装PaddlePaddle

## ***验证安装***
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

## ***如何卸载***

请使用以下命令：`pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`  卸载PaddlePaddle



