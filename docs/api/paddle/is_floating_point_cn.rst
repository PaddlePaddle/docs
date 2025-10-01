.. _cn_api_paddle_is_floating_point:

is_floating_point
-------------------------------

.. py:function:: paddle.is_floating_point(x)
判断输入 Tensor 的数据类型是否为浮点类型。

.. note::
    别名支持: 参数名 ``input`` 可替代 ``x``，如 ``is_floating_point(input=tensor_x)`` 等价于 ``is_floating_point(x=tensor_x)`` 。

参数
:::::::::

    - **x**  (Tensor) - 输入的 Tensor。别名： ``input``。

返回
:::::::::

bool，输入 tensor 的数据类型为浮点数类型则为 True，反之为 False.

代码示例
:::::::::

COPY-FROM: paddle.is_floating_point
