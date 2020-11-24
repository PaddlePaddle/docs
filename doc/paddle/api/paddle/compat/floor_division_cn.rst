.. _cn_api_paddle_compat_floor_division

floor_division
-------------------------------

.. py:function:: paddle.compat.floor_division(x, y)

等价于Python3和Python2中的除法。
在Python3中，结果为floor（x/y）的int值；在Python2中，结果为（x/y）的值。

参数
::::::::::

    - **x** (int|float) - 被除数。
    - **y** (int|float) - 除数。

返回
::::::::::
    
    x//y的除法结果
