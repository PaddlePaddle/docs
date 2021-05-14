.. _cn_api_fluid_layers_unstack:

unstack
-------------------------------

.. py:function:: paddle.unstack(x, axis=0, num=None)




该OP将单个dim为 ``D`` 的Tensor沿 ``axis`` 轴unpack为 ``num`` 个dim为 ``(D-1)`` 的Tensor

参数:
      - **x** (Tensor) – 输入x为 ``dim > 0`` 的Tensor，
      支持的数据类型: float32，float64，int32，int64。

      - **axis** (int | 可选) – 输入Tensor进行unpack运算所在的轴，axis的范围为：``[-D, D)`` ，
      如果 ``axis < 0`` ，则 :math:`axis = axis + dim(x)`，axis的默认值为0。

      - **num** (int | 可选) - axis轴的长度，一般无需设置，默认值为 ``None`` 。

返回: 长度为num的Tensor列表, 数据类型与输入Tensor相同，dim为 ``(D-1)``。


**代码示例**：

.. code-block:: python

    import paddle
    x = paddle.ones(name='x', shape=[2, 3, 5], dtype='float32')  # create a tensor with shape=[2, 3, 5]
    y = paddle.unstack(x, axis=1)  # unstack with second axis, which results 3 tensors with shape=[2, 5]






