.. _cn_api_paddle_bitwise_right_shift:

bitwise_right_shift
-------------------------------

.. py:function:: paddle.bitwise_right_shift(x, y, is_arithmetic=True, out=None, name=None)

对 Tensor ``x`` 和 ``y`` 逐元素进行 ``按位算术(或逻辑)右移`` 运算。

.. math::
        Out = X \gg Y

.. note::
    ``paddle.bitwise_right_shift`` 遵守 broadcasting，如您想了解更多，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7
参数
::::::::::::

        - **x** （Tensor）- 输入的 N-D `Tensor`，数据类型为：uint8，int8，int16，int32，int64。
        - **y** （Tensor）- 输入的 N-D `Tensor`，数据类型为：uint8，int8，int16，int32，int64。
        - **is_arithmetic** （bool） - 用于表明是否执行算术位移，True 表示算术位移，False 表示逻辑位移。默认值为 True，表示算术位移。
        - **out** （Tensor，可选）- 输出的结果 `Tensor`，是与输入数据类型相同的 N-D `Tensor`。默认值为 None，此时将创建新的 Tensor 来保存输出结果。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 ``按位算术(逻辑)右移`` 运算后的结果 ``Tensor``，数据类型与 ``x`` 相同。

代码示例
::::::::::::

COPY-FROM: paddle.bitwise_right_shift
