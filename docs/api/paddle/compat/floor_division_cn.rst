.. _cn_api_paddle_compat_floor_division:

floor_division
-------------------------------

.. py:function:: paddle.compat.floor_division(x, y)

等价于 Python3 和 Python2 中的除法。
在 Python3 中，结果为 floor（x/y）的 int 值；在 Python2 中，结果为（x/y）的值。

参数
::::::::::

    - **x** (int|float) - 被除数。
    - **y** (int|float) - 除数。

返回
::::::::::

    x//y 的除法结果
