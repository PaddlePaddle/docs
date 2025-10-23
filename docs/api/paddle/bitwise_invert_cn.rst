.. _cn_api_paddle_bitwise_invert:

bitwise_invert
-------------------------------

.. py:function:: paddle.bitwise_invert(x, out=None, name=None)

对 Tensor ``x`` 逐元素进行 ``按位取反`` 运算。

.. math::
       Out = \sim X

.. note::
       ``bitwise_invert`` 和 ``bitwise_not`` 执行相同的操作，功能等效。

参数
::::::::::::

        - **x** （Tensor）- 输入的 N-D `Tensor`，数据类型为：bool，uint8，int8，int16，int32，int64。
        - **out** （Tensor，可选）- 输出的结果 `Tensor`，是与输入数据类型相同的 N-D `Tensor`。默认值为 None，此时将创建新的 Tensor 来保存输出结果。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
 ``按位取反`` 运算后的结果 ``Tensor``，数据类型与 ``x`` 相同。

代码示例
::::::::::::

COPY-FROM: paddle.bitwise_invert
