.. _cn_api_paddle_tensor_math_pow:

pow
-------------------------------

.. py:function:: paddle.pow(x, y, name=None)



指数算子，逐元素计算 ``x`` 的 ``y`` 次幂。

.. math::

    out = x^{y}

参数
:::::::::
    - **x** （Tensor）- 多维 ``Tensor``，数据类型为 ``float16`` 、 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
    - **y** （float|int|Tensor）- 如果类型是多维 ``Tensor``，其数据类型应该和 ``x`` 相同。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
Tensor，维度和数据类型都和 ``x`` 相同。


代码示例
:::::::::

COPY-FROM: paddle.pow
