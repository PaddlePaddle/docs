==========
 安装说明
==========
本说明将指导您在64位操作系统编译和安装PaddlePaddle

PaddlePaddle目前支持的Python版本包括：Python（64 bit） 2.7-3.7

PaddlePaddle目前支持以下环境：

* *Ubuntu 14.04 /16.04 /18.04*
* *CentOS 7 / 6*
* *MacOS 10.11 / 10.12 / 10.13 / 10.14*
* *Windows7 / 8/ 10(专业版/企业版)*

请确保您的环境满足以上条件，我们默认提供的安装同时需要您的计算机拥有64位操作系统，处理器支持AVX指令集和MKL，如您有其他需求，请参考 `多版本whl包安装列表 <Tables.html/#ciwhls>`_

- 如果您希望使用 `pip <https://pypi.org/pypi/>`_ 进行安装PaddlePaddle可以直接使用以下命令:

:code:`pip install -U paddlepaddle` （CPU版本最新）

:code:`pip install -U paddlepaddle-gpu` （GPU版本最新）

注：:code:`pip install -U paddlepaddle-gpu` 命令将安装支持CUDA 9.0 cuDNN v7的PaddlePaddle，如果您的CUDA或cuDNN版本与此不同，可以参考 `这里 <https://pypi.org/project/paddlepaddle-gpu/#history>`_ 了解其他CUDA/cuDNN版本所适用的安装命令

如果您希望通过 ``pip`` 方式安装老版本的PaddlePaddle，您可以使用如下命令：

:code:`pip install -U paddlepaddle==[PaddlePaddle版本号]` (CPU版，具体版本号请参考 `这里 <https://pypi.org/project/paddlepaddle/#history>`_ )

:code:`pip install -U paddlepaddle-gpu==[PaddlePaddle版本号]` (GPU版,具体版本号请参考 `这里 <https://pypi.org/project/paddlepaddle-gpu/#history>`_ )

- 如果您希望使用 `docker <https://www.docker.com>`_ 安装PaddlePaddle，可以使用以下命令:

:code:`docker run --name [Name of container] -it -v $PWD:/paddle hub.baidubce.com/paddlepaddle/paddle:[docker版本号] /bin/bash`

如果您的机器不在中国大陆地区，可以直接从DockerHub拉取镜像

:code:`docker run --name [Name of container] -it -v $PWD:/paddle paddlepaddle/paddle:[docker版本号] /bin/bash`

	其中[docker版本号]请查阅 `DockerHub <https://hub.docker.com/r/paddlepaddle/paddle/tags/>`_
	
- 如果您有开发PaddlePaddle的需求，请参考：`从源码编译 <compile/fromsource.html>`_

..	toctree::
	:hidden:

	install_Ubuntu.md
	install_CentOS.md
	install_MacOS.md
	install_Windows.md
	install_Docker.md
	compile/fromsource.rst
	Tables.md
