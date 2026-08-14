.. _cn_api_paddle_greater_than:

greater_than
-------------------------------
.. py:function:: paddle.greater_than(x, y, name=None, *, out=None)


逐元素返回 :math:`x > y` 的真值；对应位置的 ``x`` 大于 ``y`` 时返回 True，否则返回 False。使用重载算子 ``>`` 可以获得相同的计算效果。

.. note::
    输出的结果不返回梯度。

参数
:::::::::
    - **x** (Tensor) - 用于比较的第一个输入 Tensor，支持的数据类型包括 bool、bfloat16、float16、float32、float64、uint8、int8、int16、int32、int64、complex64、complex128。别名 ``input``。
    - **y** (Tensor) - 用于比较的第二个输入 Tensor，支持的数据类型包括 bool、bfloat16、float16、float32、float64、uint8、int8、int16、int32、int64、complex64、complex128。别名 ``other``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


关键字参数
:::::::::::
    - **out** (Tensor，可选) - 输出 Tensor，若不为 ``None``，计算结果将保存在该 Tensor 中，默认值为 ``None``。

返回
:::::::::
Tensor，输出结果，shape 和输入一致，Tensor 数据类型为 bool。


代码示例
:::::::::

COPY-FROM: paddle.greater_than
