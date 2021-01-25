.. _cn_api_fluid_layers_multiply:

multiply
-------------------------------

.. py:function:: paddle.multiply(x, y, name=None)




该OP是逐元素相乘算子，输入 ``x`` 与输入 ``y`` 逐元素相乘，并将各个位置的输出元素保存到返回结果中。

等式是：

.. math::
        Out = X \odot Y

- :math:`X` ：多维Tensor。
- :math:`Y` ：维度必须小于等于X维度的Tensor。

对于这个运算算子有2种情况：
        1. :math:`Y` 的 ``shape`` 与 :math:`X` 相同。
        2. :math:`Y` 的 ``shape`` 是 :math:`X` 的连续子序列。
        3. 输入 ``x`` 与输入 ``y`` 必须可以广播为相同形状, 关于广播规则，请参考 :ref:`cn_user_guide_broadcasting` 。

参数
:::::::::

        - **x** （Tensor）- 多维 ``Tensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
        - **y** （Tensor）- 多维 ``Tensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64`` 。
        - **name** （string，可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。


返回
:::::::::
   ``Tensor``，存储运算后的结果。如果x和y有不同的shape且是可以广播的，返回Tensor的shape是x和y经过广播后的shape。如果x和y有相同的shape，返回Tensor的shape与x，y相同。


代码示例
:::::::::

..  code-block:: python

    import paddle

    x = paddle.to_tensor([[1, 2], [3, 4]])
    y = paddle.to_tensor([[5, 6], [7, 8]])
    res = paddle.multiply(x, y)
    print(res) # [[5, 12], [21, 32]]

    x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
    y = paddle.to_tensor([2])
    res = paddle.multiply(x, y)
    print(res) # [[2, 4, 6], [2, 4, 6]]]







