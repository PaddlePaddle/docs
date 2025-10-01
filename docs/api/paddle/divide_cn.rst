.. _cn_api_paddle_divide:

divide
-------------------------------

.. py:function:: paddle.divide(x, y, name=None, *, rounding_mode=None, out=None)

逐元素相除算子，输入 ``x`` 与输入 ``y`` 逐元素相除，并将各个位置的输出元素保存到返回结果中。

.. note::
    别名支持: 参数名 ``input`` 可替代 ``x``，参数名 ``other`` 可替代 ``y`` ，如 ``divide(input=tensor_x, other=tensor_y, ...)`` 等价于 ``divide(x=tensor_x, y=tensor_y, ...)`` 。
    输入 ``x`` 与输入 ``y`` 必须和广播为相同形状，关于广播规则，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

等式为：

.. math::
        Out = X / Y

- :math:`X`：多维 Tensor。
- :math:`Y`：多维 Tensor。

参数
:::::::::
        - **x** (Tensor) - 多维 Tensor。数据类型为 bool、bfloat16、float16、float32、float64、int8、int16、int32、int64、uint8、complex64、complex128。
          ``别名: input``
        - **y** (Tensor) - 多维 Tensor。数据类型为 bool、bfloat16、float16、float32、float64、int8、int16、int32、int64、uint8、complex64、complex128。
          ``别名: other``
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
        - **rounding_mode** (str，可选) - 指定舍入模式。可选值为 ``None``、``"trunc"`` 或 ``"floor"``。如果 ``rounding_mode=None``，则不进行舍入操作；如果 ``rounding_mode="trunc"``，则向零截断；如果 ``rounding_mode="floor"``，则向负无穷舍入。
        - **out** (Tensor，可选) - 输出 Tensor，默认值为 None。


返回
:::::::::

   ``Tensor``，存储运算后的结果。如果 x 和 y 有不同的 shape 且是可以广播的，返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。如果 x 和 y 有相同的 shape，返回 Tensor 的 shape 与 x，y 相同。



代码示例
:::::::::

COPY-FROM: paddle.divide
