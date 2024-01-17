.. _cn_api_paddle_version_xpu:

xpu
-------------------------------

.. py:function:: paddle.version.xpu()

获取 paddle 安装包编译时使用的 XPU 版本号。


返回
::::::::::

若 paddle wheel 包为 XPU 版本，则返回 paddle wheel 包编译时使用的 XPU 的版本信息；若 paddle wheel 包为 非 XPU 版本，则返回 ``False`` 。

代码示例：
::::::::::

COPY-FROM: paddle.version.xpu
