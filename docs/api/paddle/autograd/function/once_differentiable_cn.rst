.. _cn_api_paddle_autograd_function_once_differentiable:

once_differentiable
-------------------------------

.. py:function:: paddle.autograd.function.once_differentiable(backward)

``paddle.autograd.py_layer.once_differentiable`` 的别名，用作 ``PyLayer.backward`` 方法的装饰器，使该反向方法本身不可再次求导。

参数
:::::::::

    - **backward** (Callable) - 要包装的 ``PyLayer.backward`` 函数。

返回
:::::::::

Callable，包装后的反向函数。
