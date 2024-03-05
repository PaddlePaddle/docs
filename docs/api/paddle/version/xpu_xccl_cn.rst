.. _cn_api_paddle_version_xpu_xccl:

xpu_xccl
-------------------------------

.. py:function:: paddle.version.xpu_xccl()

获取 paddle 安装包编译时使用的 XPU xccl 版本号。

返回
::::::::::

若 paddle wheel 包为 XPU 版本，则返回 paddle wheel 包编译时使用的 XPU xccl 版本信息；若 paddle wheel 包为 非 XPU 版本，则返回 ``False`` 。

代码示例：
::::::::::

COPY-FROM: paddle.version.xpu_xccl
