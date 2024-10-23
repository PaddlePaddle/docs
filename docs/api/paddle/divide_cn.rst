.. _cn_api_paddle_divide:

divide
-------------------------------

.. py:function:: paddle.divide(x, y, name=None)

逐元素相除算子，输入 ``x`` 与输入 ``y`` 逐元素相除，并将各个位置的输出元素保存到返回结果中。

.. note::
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
        - **y** (Tensor) - 多维 Tensor。数据类型为 bool、bfloat16、float16、float32、float64、int8、int16、int32、int64、uint8、complex64、complex128。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::

   ``Tensor``，存储运算后的结果。如果 x 和 y 有不同的 shape 且是可以广播的，返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。如果 x 和 y 有相同的 shape，返回 Tensor 的 shape 与 x，y 相同。



代码示例
:::::::::

COPY-FROM: paddle.divide
