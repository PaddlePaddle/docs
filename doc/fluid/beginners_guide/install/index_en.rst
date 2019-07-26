======================
 Installation Manuals
======================

The manuals will guide you to install and build PaddlePaddle on your 64-bit desktop or laptop.

The versions of Python currently supported: Python 2.7-3.7

PaddlePaddle currently supports the following environments：

* *Ubuntu 14.04 /16.04 /18.04*
* *CentOS 7 / 6*
* *MacOS 10.11 / 10.12 / 10.13 / 10.14*
* *Windows7 / 8/ 10(Pro/Enterprise)*


Please make sure your environment meets the conditions above.
And the installation assumes your computer possesses 64-bit operating system, and AVX instruction set is supported by the processor, otherwise you should use the version of :code:`no_avx` in `whl package list - Dev <Tables_en.html/#ciwhls>`_ .


- If you are planning to use  `pip <https://pypi.org/pypi/>`_ to install PaddlePaddle, please type the following commands directly:

:code:`pip install paddlepaddle` （latest CPU version of PaddlePaddle）

:code:`pip install paddlepaddle-gpu` （latest GPU version of PaddlePaddle）

:code:`pip install paddlepaddle==[pip version]`

	where [pip version] can be looked up in `PyPi.org <https://pypi.org/search/?q=PaddlePaddle>`_

- If you are planning to use `docker <https://www.docker.com>`_ to install PaddlePaddle, please type the following commands directly:

:code:`docker run --name [Name of container] -it -v $PWD:/paddle paddlepaddle/paddle:[docker version] /bin/bash`

    where [docker version] can be looked up in `DockerHub <https://hub.docker.com/r/paddlepaddle/paddle/tags/>`_

..	toctree::
	:hidden:

	install_Ubuntu_en.md
	install_CentOS_en.md
	install_MacOS_en.md
	install_Windows_en.md
	compile/fromsource_en.rst
	Tables_en.md
