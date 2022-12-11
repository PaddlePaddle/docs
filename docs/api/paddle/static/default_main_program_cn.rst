.. _cn_api_fluid_default_main_program:

default_main_program
-------------------------------

.. py:function:: paddle.static.default_main_program()

此接口可以获取当前用于存储 OP 和 Tensor 描述信息的 ``default main program``。

例如 ``z = paddle.add(x, y)`` 会创建新 ``Op`` 和 tensor ``z``，这些变量会被记录在 ``default main program`` 中。

``default main program`` 是许多编程接口中 Program 参数的默认值。例如对于 ``Executor.run()`` 如果用户没有传入 Program 参数，会默认使用 ``default main program`` 。

可以使用 :ref:`cn_api_fluid_program_guard` 来切换 ``default main program``。

返回
:::::::::

 :ref:`cn_api_fluid_Program`，当前默认用于存储 OP 和 Tensor 描述的 Program。


代码示例
:::::::::

COPY-FROM: paddle.static.default_main_program
