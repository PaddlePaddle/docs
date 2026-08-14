.. _cn_api_paddle_utils_cpp_extension_get_build_directory:

get_build_directory
-------------------------------

.. py:function:: paddle.utils.cpp_extension.get_build_directory(verbose=False)

此接口返回编译自定义 OP 时生成动态链接库所在的 build 目录路径。此目录可以通过 ``export PADDLE_EXTENSION_DIR=XXX`` 来设置。若未设定，则默认使用 ``~/.cache/paddle_extensions`` 作为 build 目录。

参数
::::::::::::

    - **verbose** (bool，可选) - 当未设置 ``PADDLE_EXTENSION_DIR`` 时，是否打印所使用的默认目录信息。默认值为 False。


返回
::::::::::::
编译自定义 OP 的 build 目录路径。

代码示例
::::::::::::

COPY-FROM: paddle.utils.cpp_extension.get_build_directory
