.. _cn_api_paddle_multiply:

multiply
-------------------------------

.. py:function:: paddle.multiply(x, y, name=None)




逐元素相乘算子，输入 ``x`` 与输入 ``y`` 逐元素相乘，并将各个位置的输出元素保存到返回结果中。

等式是：

.. math::
        Out = X \odot Y

- :math:`X`：多维 Tensor。
- :math:`Y`：维度必须小于等于 X 维度的 Tensor。

对于这个运算算子有 2 种情况：

        1. :math:`Y` 的 ``shape`` 与 :math:`X` 相同。
        2. :math:`Y` 的 ``shape`` 是 :math:`X` 的连续子序列。
        3. 输入 ``x`` 与输入 ``y`` 必须可以广播为相同形状，关于广播规则，请参见 `Tensor 介绍`_ .

        .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
:::::::::

        - **x** （Tensor）- 多维 ``Tensor``。数据类型为 ``bfloat16`` 、 ``float16`` 、 ``float32`` 、 ``float64`` 、 ``int32`` 、 ``int64``、 ``bool``、 ``complex64`` 或  ``complex128``。
        - **y** （Tensor）- 多维 ``Tensor``。数据类型为 ``bfloat16`` 、 ``float16`` 、 ``float32`` 、 ``float64`` 、 ``int32`` 、 ``int64``、 ``bool``、 ``complex64`` 或  ``complex128``。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
   ``Tensor``，存储运算后的结果。如果 x 和 y 有不同的 shape 且是可以广播的，返回 Tensor 的 shape 是 x 和 y 经过广播后的 shape。如果 x 和 y 有相同的 shape，返回 Tensor 的 shape 与 x，y 相同。


代码示例
:::::::::

COPY-FROM: paddle.multiply
