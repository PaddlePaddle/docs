========
概述
========

您可以使用我们提供的安装包，或使用源代码，安装PaddlePaddle.

pip安装
=============================
PaddlePaddle Python API 依赖Python 2.7版本
建议您使用`pip <https://pypi.org/project/pip/>`
安装，它是Linux系统下最简单的安装方式。

Ubuntu/CentOS
=============================
执行下面的命令即可在当前机器上安装PaddlePaddle的运行时环境，并自动下载安装依赖软件。

  .. code-block:: bash

     pip install paddlepaddle

当前的默认版本为0.13.0，cpu_avx_openblas，您可以通过指定版本号来安装其它版本，例如:

  .. code-block:: bash

      pip install paddlepaddle==0.12.0


如果需要安装支持GPU的版本（cuda9.0_cudnn7_avx_openblas），需要执行：

  .. code-block:: bash

     pip install paddlepaddle-gpu
