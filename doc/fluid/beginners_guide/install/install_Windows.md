***

# **Windows下安装**

本文将介绍如何在Windows系统下安装PaddlePaddle，您的计算机需要满足以下要求：

* *64位台式机或笔记本电脑*

* *Windows 7/8 ，Windows 10 专业版/企业版*

注：

* 当前版本暂不支持NCCL，分布式等相关功能

## 安装步骤

### ***使用pip安装***

* 对Python版本的要求

我们支持[Python官方提供](https://www.python.org/downloads/)的Python2.7.15，Python3.5.x，Python3.6.x，Python3.7.x

* 对pip版本的要求

您的pip或pip3版本号需不低于9.0.1。pip与python版本是对应的, 如果是python2.7, 建议使用`pip`命令; 如果是python3.x, 则建议使用`pip3`命令。

* 开始安装

* ***CPU版本的PaddlePaddle***:
执行如下命令：`pip install paddlepaddle`(python2.7) 或 `pip3 install paddlepaddle`(python3.x) 安装PaddlePaddle

* ***GPU版本的PaddlePaddle***:
执行如下命令：`pip install paddlepaddle-gpu`(python2.7) 或 `pip3 install paddlepaddle-gpu`(python3.x) 安装PaddlePaddle

## ***验证安装***
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

## ***如何卸载***

* ***CPU版本的PaddlePaddle***:
请使用以下命令：`pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`  卸载PaddlePaddle

* ***GPU版本的PaddlePaddle***:
请使用以下命令：`pip uninstall paddlepaddle-gpu` 或 `pip3 uninstall paddlepaddle-gpu`  卸载PaddlePaddle

