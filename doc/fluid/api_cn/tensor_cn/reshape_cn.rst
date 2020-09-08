.. _cn_api_tensor_cn_reshape:

reshape
-------------------------------

.. py:function::  paddle.reshape(x, shape, name=None)

:alias_main: paddle.reshape
:alias: paddle.reshape,paddle.tensor.reshape,paddle.tensor.manipulation.reshape


该OP在保持输入 ``x`` 数据不变的情况下，改变 ``x`` 的形状。

在指定目标shape时存在一些技巧：

.. code-block:: text

  1. -1 表示这个维度的值是从x的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。
  2. 0 表示实际的维数是从x的对应维数中复制出来的，因此shape中0的索引值不能超过x的维度。


这里有一些例子来解释它们：

.. code-block:: text

  1. 给定一个形状为[2,4,6]的三维张量x，目标形状为[6,8]，则将x变换为形状为[6,8]的2-D张量，且x的数据保持不变。
  2. 给定一个形状为[2,4,6]的三维张量x，目标形状为[2,3,-1,2]，则将x变换为形状为[2,3,4,2]的4-D张量，且x的数据保持不变。在这种情况下，目标形状的一个维度被设置为-1，这个维度的值是从x的元素总数和剩余维度推断出来的。
  3. 给定一个形状为[2,4,6]的三维张量x，目标形状为[-1,0,3,2]，则将x变换为形状为[2,4,3,2]的4-D张量，且x的数据保持不变。在这种情况下，0对应位置的维度值将从x的对应维数中复制,-1对应位置的维度值由x的元素总数和剩余维度推断出来。


参数：
  - **x** （Tensor）- 多维 ``Tensor``，数据类型为 ``float32``，``float64``，``int32``，或 ``int64``。
  - **shape** （list|tuple|Tensor）- 数据类型是 ``int32`` 。定义目标形状。目标形状最多只能有一个维度为-1。如果 ``shape`` 的类型是 list 或 tuple, 它的元素可以是整数或者形状为[1]的 ``Tensor`` 或 ``LoDTensor``。如果 ``shape`` 的类型是 ``Tensor``，则是1-D的 ``Tensor`` 或 ``LoDTensor``。
  - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置。默认值： ``None``。

返回：
:::::::::
``Tensor``, 改变形状后的 ``Tensor``，数据类型与 ``x`` 相同。


**代码示例**

.. code-block:: python

    import numpy as np
    import paddle

    paddle.disable_static()

    data = np.random.random([2, 4, 6]).astype("float32")
    x = paddle.to_tensor(data)

    positive_four = paddle.fill_constant([1], "int32", 4)

    out_1 = paddle.reshape(x, [-1, 0, 3, 2])
    # the shape of out_1 is [2,4,3,2].

    out_2 = paddle.reshape(x, shape=[positive_four, 12])
    # the shape of out_2 is [4, 12].

    shape_tensor = paddle.to_tensor(np.array([8, 6]).astype("int32"))
    out_3 = paddle.reshape(x, shape=shape_tensor)
    # the shape of out_2 is [8, 6].
