.. _cn_api_fluid_layers_multiply:

multiply
-------------------------------

.. py:function:: paddle.multiply(x, y, axis=-1, name=None)

:alias_main: paddle.multiply
:alias: paddle.multiply, paddle.tensor.multiply, paddle.tensor.math.multiply



该OP是逐元素相乘算子，输入 ``x`` 与输入 ``y`` 逐元素相乘，并将各个位置的输出元素保存到返回结果中。

等式是：

.. math::
        Out = X \odot Y

- :math:`X` ：多维Tensor。
- :math:`Y` ：维度必须小于等于X维度的Tensor。

对于这个运算算子有2种情况：
        1. :math:`Y` 的 ``shape`` 与 :math:`X` 相同。
        2. :math:`Y` 的 ``shape`` 是 :math:`X` 的连续子序列。

对于情况2:
        1. 用 :math:`Y` 匹配 :math:`X` 的形状（shape），其中 ``axis`` 是 :math:`Y` 在 :math:`X` 上的起始维度的位置。
        2. 如果 ``axis`` 为-1（默认值），则 :math:`axis= rank(X)-rank(Y)` 。
        3. 考虑到子序列， :math:`Y` 的大小为1的尾部维度将被忽略，例如shape（Y）=（2,1）=>（2）。

例如：

..  code-block:: text

        shape(X) = (2, 3, 4, 5), shape(Y) = (,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (5,)
        shape(X) = (2, 3, 4, 5), shape(Y) = (4, 5), with axis=-1(default) or axis=2
        shape(X) = (2, 3, 4, 5), shape(Y) = (3, 4), with axis=1
        shape(X) = (2, 3, 4, 5), shape(Y) = (2), with axis=0
        shape(X) = (2, 3, 4, 5), shape(Y) = (2, 1), with axis=0

参数：
        - **x** （Variable）- 多维 ``Tensor`` 或 ``LoDTensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64``。
        - **y** （Variable）- 多维 ``Tensor`` 或 ``LoDTensor`` 。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 或  ``int64``。
        - **axis** （int32，可选）-  ``y`` 的维度对应到 ``x`` 维度上时的索引。默认值为 -1。
        - **name** （string，可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。


返回：        维度与 ``x`` 相同的 ``Tensor`` 或 ``LoDTensor`` ，数据类型与 ``x`` 相同。

返回类型：        Variable。

**代码示例 1**

..  code-block:: python

    import paddle
    import numpy as np
    paddle.enable_imperative()
    x_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y_data = np.array([[5, 6], [7, 8]], dtype=np.float32)
    x = paddle.imperative.to_variable(x_data)
    y = paddle.imperative.to_variable(y_data)
    res = paddle.multiply(x, y)
    print(res.numpy()) # [[5, 12], [21, 32]]
    x_data = np.array([[[1, 2, 3], [1, 2, 3]]], dtype=np.float32)
    y_data = np.array([1, 2], dtype=np.float32)
    x = paddle.imperative.to_variable(x_data)
    y = paddle.imperative.to_variable(y_data)
    res = paddle.multiply(x, y, axis=1)
    print(res.numpy()) # [[[1, 2, 3], [2, 4, 6]]]







