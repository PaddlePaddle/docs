.. _cn_api_paddle_amp_debugging_check_layer_numerics:

check_layer_numerics
-------------------------------

.. py:function:: paddle.amp.debugging.check_layer_numerics(func)

这个装饰器用于检查层的输入和输出数据的数值。


参数
:::::::::

- **func** (callable) – 将要被装饰的函数。

返回
:::::::::
一个被装饰后的函数（callable）。这个新的函数会在原来的函数基础上加上数值检查的功能。


代码示例
::::::::::::

COPY-FROM: paddle.amp.debugging.check_layer_numerics
