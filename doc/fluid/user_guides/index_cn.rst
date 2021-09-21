########
典型案例
########

..  todo::

如果您已经掌握了快速上手阶段的内容，期望可以针对实际问题建模、搭建自己网络，本模块提供了一些 Paddle 的具体典型案例供您参考：

本章文档将指导您如何使用PaddlePaddle完成基础的深度学习任务

本章文档涉及大量了深度学习基础知识，也介绍了如何使用PaddlePaddle实现这些内容，请参阅以下说明了解如何使用：

内容简介
======================


    - `简单案例 <../user_guides/simple_case/index_cn.html>`_ ：介绍了 Paddle 的基本案例

    - `计算机视觉 <../user_guides/cv_case/index_cn.html>`_ ：介绍使用 Paddle 解决计算机视觉领域的案例

    - `自然语言处理 <../user_guides/nlp_case/index_cn.html>`_： 介绍使用 Paddle 实现自然语言处理方向的案例

    - `推荐 <../user_guides/rec_case/index_cn.html>`_：介绍如何使用 Paddle 完成推荐领域任务的案例

    - `工具组件 <../user_guides/tools/index_cn.html>`_：介绍在 Paddle 工具组件的使用案例

..  toctree::
    :hidden:

    simple_case/index_cn.rst
    cv_case/index_cn.rst
    nlp_case/index_cn.rst
    rec_case/index_cn.rst
    tools/index_cn.rst
	

我们把Jupyter、PaddlePaddle、以及各种被依赖的软件都打包进一个Docker image了。所以您不需要自己来安装各种软件，只需要安装Docker即可。对于各种Linux发行版，请参考 https://www.docker.com 。如果您使用 `Windows <https://www.docker.com/docker-windows>`_ 或者 `Mac <https://www.docker.com/docker-mac>`_，可以考虑 `给Docker更多内存和CPU资源 <http://stackoverflow.com/a/39720010/724872>`_ 。

使用方法
======================

本书默认使用CPU训练，若是要使用GPU训练，使用步骤会稍有变化,请参考下文“使用GPU训练”

使用CPU训练
>>>>>>>>>>>>

只需要在命令行窗口里运行：

..  code-block:: shell

	docker run -d -p 8888:8888 paddlepaddle/book

即可从DockerHub.com下载和运行本书的Docker image。阅读和在线编辑本书请在浏览器里访问 http://localhost:8888

使用GPU训练
>>>>>>>>>>>>>

为了保证GPU驱动能够在镜像里面正常运行，我们推荐使用 `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ 来运行镜像。请先安装nvidia-docker，之后请运行：

::

	nvidia-docker run -d -p 8888:8888 paddlepaddle/book:latest-gpu


还需要将以下代码

..  code-block:: python

	use_cuda = False


改成：

..  code-block:: python

	use_cuda = True

贡献新章节
=============

您要是能贡献新的章节那就太好了！请发Pull Requests把您写的章节加入到 :code:`pending` 下面的一个子目录里。当这一章稳定下来，我们一起把您的目录挪到根目录。

为了写作、运行、调试，您需要安装Python 2.x和Go >1.5, 并可以用 `脚本程序 <https://github.com/PaddlePaddle/book/blob/develop/.tools/convert-markdown-into-ipynb-and-test.sh>`_ 来生成新的Docker image。

**Please Note:** We also provide `English Readme <https://github.com/PaddlePaddle/book/blob/develop/README.md>`_ for PaddlePaddle book

