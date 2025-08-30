.. _cn_api_paddle_bitwise_or:

bitwise_or
-------------------------------

.. py:function:: paddle.bitwise_or(x, y, out=None, name=None)

对 Tensor ``x`` 和 ``y`` 逐元素进行 ``按位或`` 运算。

.. math::
       Out = X | Y

.. note::
    ``paddle.bitwise_or`` 遵守 broadcasting，如您想了解更多，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

.. note::
    别名支持: 参数名 ``input`` 可替代 ``x`` 和 ``other`` 可替代 ``y``，如 ``input=tensor_x`` 等价于 ``x=tensor_x``， ``other=tensor_y`` 等价于 ``y=tensor_y``。

参数
::::::::::::

        - **x** （Tensor）- 输入的 N-D `Tensor`，数据类型为：bool，uint8，int8，int16，int32，int64。
        - **input** - ``x`` 的别名，行为完全一致。
        - **y** （Tensor）- 输入的 N-D `Tensor`，数据类型为：bool，uint8，int8，int16，int32，int64。
        - **other** - ``y`` 的别名，行为完全一致。
        - **out** （Tensor，可选）- 输出的结果 `Tensor`，是与输入数据类型相同的 N-D `Tensor`。默认值为 None，此时将创建新的 Tensor 来保存输出结果。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 ``按位或`` 运算后的结果 ``Tensor``，数据类型与 ``x`` 相同。

代码示例
::::::::::::

COPY-FROM: paddle.bitwise_or
