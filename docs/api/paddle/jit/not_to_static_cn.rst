.. _cn_api_paddle_jit_not_to_static:

not_to_static
-------------------------------

.. py:decorator:: paddle.jit.not_to_static

被本装饰器装饰的函数在动转静过程不会进行动转静代码转写。

参数
:::::::::
    - **function** (callable)：装饰的函数。

返回
:::::::::
callable，一个在动转静过程不会进行代码转写的函数。

示例代码
:::::::::

COPY-FROM: paddle.jit.not_to_static
