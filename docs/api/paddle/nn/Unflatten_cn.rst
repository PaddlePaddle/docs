.. _cn_api_nn_Unflatten:

Unflatten
-------------------------------

.. py:function:: paddle.nn.Unflatten(shape, axis, name=None)



构造一个 ``Unflatten`` 类的可调用对象。更多信息请参见代码示例。"将输入 Tensor 沿指定轴 axis 上的维度展成 shape 形状"。


参数
::::::::::::

    - **shape** (list|tuple|Tensor) - 在指定轴上将该维度展成 ``shape``， 其中 ``shape`` 最多包含一个 -1，如果输入 ``shape`` 不包含 -1 ，则乘积应该等于 ``x.shape[axis]`` 大小。 数据类型为 ``int``。如果 ``shape`` 的类型是 ``list`` 或 ``tuple``，它的元素可以是整数或者形状为[]的 ``Tensor``。如果 ``shape`` 的类型是 ``Tensor``，则是 1-D 的 ``Tensor``。
    - **axis** (int) - 要展开的轴，作为 ``x.shape`` 的索引。

返回
::::::::::::
  无。


代码示例
::::::::::::

COPY-FROM: paddle.nn.Unflatten
