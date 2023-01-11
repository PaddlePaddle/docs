.. _cn_api_paddle_jit_ignore_module:

ignore_module
-------------------------------

.. py:function:: paddle.jit.ignore_module(modules)

本接口可以自定义增加动转静过程中忽略转写的模块，目前默认忽略转写的模块有 collections、 pdb、 copy、 inspect、 re、 numpy、 logging、 six。

参数
::::::::::::

    - **modules** (List[Any]) - 动转静过程中要增加的忽略转写的模块列表

代码示例
::::::::::::

COPY-FROM: paddle.jit.ignore_module
