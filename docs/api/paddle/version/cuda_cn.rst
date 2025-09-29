.. _cn_api_paddle_version_cuda:

cuda
-------------------------------

.. py:function:: paddle.version.cuda()

获取 paddle 安装包编译时使用的 CUDA 版本号。


返回
::::::::::

若 paddle wheel 包为 GPU 版本，则返回 paddle wheel 包编译时使用的 CUDA 的版本信息；若 paddle wheel 包为 CPU 版本，则返回 ``False`` 。

代码示例：
::::::::::

COPY-FROM: paddle.version.cuda
