

# 沐曦 曦云 C500 基于 PaddlePaddle 框架的使用指南

## 一、环境准备

### 环境说明

* 本教程介绍如何基于沐曦 曦云 C500 进行安装使用

* 考虑到环境差异性，我们推荐使用教程提供的标准镜像完成环境准备：

  * x86_64 镜像链接：您可以联系 MetaX 或访问 https://sw-download.metax-tech.com 获取对应的镜像文件。

  * 镜像中已经默认安装了沐曦 MACA 软件栈


### 环境安装

1. 安装 PaddlePaddle

*该命令会自动安装飞桨主框架每日自动构建的 nightly-build 版本*

```shell
python -m pip install  --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```

2. 安装 CustomDevice

*该命令会自动安装飞桨 Custom Device 每日自动构建的 nightly-build 版本*

```shell
python -m pip install --pre paddle-metax-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/maca/
```
