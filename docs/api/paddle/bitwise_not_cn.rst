.. _cn_api_paddle_bitwise_not:

bitwise_not
-------------------------------

.. py:function:: paddle.bitwise_not(x, out=None, name=None)

对 Tensor ``x`` 逐元素进行 ``按位取反`` 运算。

.. math::
       Out = \sim X

.. note::
    ``paddle.bitwise_not`` 遵守 broadcasting，如您想了解更多，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
::::::::::::

        - **x** （Tensor）- 输入的 N-D `Tensor`，数据类型为：bool，uint8，int8，int16，int32，int64。
        - **out** （Tensor，可选）- 输出的结果 `Tensor`，是与输入数据类型相同的 N-D `Tensor`。默认值为 None，此时将创建新的 Tensor 来保存输出结果。

返回
::::::::::::
 ``按位取反`` 运算后的结果 ``Tensor``，数据类型与 ``x`` 相同。

代码示例
::::::::::::

COPY-FROM: paddle.bitwise_not
