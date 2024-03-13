.. _cn_api_paddle_version_nccl:

nccl
-------------------------------

.. py:function:: paddle.version.nccl()

获取 paddle 安装包编译时使用的 nccl 版本号。


返回
:::::::::

若 paddle wheel 包为 GPU 版本，则返回 paddle wheel 包编译时使用的 nccl 的版本信息；若 paddle wheel 包为 CPU 版本，则返回 ``False`` 。

代码示例：
::::::::::

COPY-FROM: paddle.version.nccl
