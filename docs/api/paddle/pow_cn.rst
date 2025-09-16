.. _cn_api_paddle_pow:

pow
-------------------------------

.. py:function:: paddle.pow(x, y, name=None, *, out=None)



指数算子，逐元素计算 ``x`` 的 ``y`` 次幂。

.. math::

    out = x^{y}

.. note::
    别名支持: 参数名 ``input`` 可替代 ``x``，参数名 ``exponent`` 可替代 ``y`` ，如 ``pow(input=2, exponent=1.1)`` 等价于 ``pow(x=2, y=1.1)`` 。

参数
:::::::::
    - **x** （Tensor）- 多维 ``Tensor``，数据类型为 ``bfloat16`` 、 ``float16`` 、 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。别名： ``input``。
    - **y** （float|int|Tensor）- 如果类型是多维 ``Tensor``，其数据类型应该和 ``x`` 相同。别名： ``exponent``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::

    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
Tensor，维度和数据类型都和 ``x`` 相同。


代码示例
:::::::::

COPY-FROM: paddle.pow
