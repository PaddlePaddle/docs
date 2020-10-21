.. _cn_api_tensor_remainder:

remainder
-------------------------------

.. py:function:: paddle.remainder(x, y, name=None)

该OP是逐元素取模算子，输入 ``x`` 与输入 ``y`` 逐元素取模，并将各个位置的输出元素保存到返回结果中。
输入 ``x`` 与输入 ``y`` 必须可以广播为相同形状, 关于广播规则，请参考 :ref:`use_guide_broadcasting`

等式为：

.. math::
        Out = X \% Y

- :math:`X` ：多维Tensor。
- :math:`Y` ：多维Tensor。

参数：
        - x（Tensor）- 多维Tensor。数据类型为float32 、float64、int32或int64。
        - y（Tensor）- 多维Tensor。数据类型为float32 、float64、int32或int64。
        - name（str，可选）- 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。


返回：   多维 Tensor， 数据类型与 ``x`` 相同，维度为广播后的形状。

返回类型：        Tensor


**代码示例**

..  code-block:: python

        import paddle
        import numpy as np
        paddle.disable_static()
        np_x = np.array([2, 3, 8, 7])
        np_y = np.array([1, 5, 3, 3])
        x = paddle.to_tensor(np_x)
        y = paddle.to_tensor(np_y)
        z = paddle.remainder(x, y)
        print(z.numpy())  # [0, 3, 2, 1]