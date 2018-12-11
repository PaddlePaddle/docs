==========
 安装说明
==========
本说明将指导您在64位台式机或笔记本电脑上编译和安装PaddlePaddle

PaddlePaddle目前支持的Python版本包括：Python 2.7-3.7

PaddlePaddle目前支持以下环境：

* *Ubuntu 14.04 /16.04 /18.04*
* *CentOS 7 / 6*
* *MacOS 10.11 / 10.12 / 10.13 / 10.14*
* *Windows7 / 8/ 10(专业版/企业版)*


请确保您的环境满足以上条件，我们默认提供的安装同时需要您的计算机拥有64位操作系统，处理器支持AVX指令集，否则请选择 `多版本whl包安装列表 <Tables.html/#ciwhls>`_  中 :code:`no_avx` 的版本

- 如果您希望使用 `pip <https://pypi.org/pypi/>`_ 进行安装PaddlePaddle可以直接使用以下命令:

:code:`pip install paddlepaddle` （CPU版本最新）

:code:`pip install paddlepaddle-gpu` （GPU版本最新）

:code:`pip install paddlepaddle==[pip版本号]`

	其中[pip版本号]请查阅 `PyPi.org <https://pypi.org/search/?q=PaddlePaddle>`_

- 如果您希望使用 `docker <https://www.docker.com>`_ 安装PaddlePaddle可以直接使用以下命令:
:code:`docker run --name [Name of container] -it -v $PWD:/paddle hub.baidubce.com/paddlepaddle/paddle:[docker版本号] /bin/bash`

	其中[docker版本号]请查阅 `DockerHub <https://hub.docker.com/r/paddlepaddle/paddle/tags/>`_

..	toctree::
	:hidden:

	install_Ubuntu.md
	install_CentOS.md
	install_MacOS.md
	install_Windows.md
	compile/fromsource.rst
	FAQ.md
	Tables.md
