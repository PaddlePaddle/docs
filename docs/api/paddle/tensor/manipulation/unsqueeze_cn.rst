.. _cn_api_paddle_tensor_unsqueeze:

unsqueeze
-------------------------------

.. py:function:: paddle.unsqueeze(x, axis, name=None)

该OP向输入Tensor的Shape中一个或多个位置（axis）插入尺寸为1的维度。

请注意，在动态图模式下，输出Tensor将与输入Tensor共享数据，并且没有Tensor数据拷贝的过程。
如果不希望输入与输出共享数据，请使用 `Tensor.clone` ，例如 `unsqueeze_clone_x = x.unsqueeze(-1).clone()` 。

参数
:::::::::
        - **x** (Tensor)- 输入的 `Tensor` ，数据类型为：float32、float64、bool、int8、int32、int64。
        - **axis** (int|list|tuple|Tensor) - 表示要插入维度的位置。数据类型是 int32 。如果 axis 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 Tensor 。如果 axes 的类型是 Tensor，则是1-D Tensor。如果 axis 是负数，则 axis=axis+ndim(x)+1 。
        - **name** （str，可选）- 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
Tensor,扩展维度后的多维Tensor，数据类型与输入Tensor一致。

代码示例
:::::::::

.. code-block:: python

    import paddle

    x = paddle.rand([5, 10])
    print(x.shape)  # [5, 10]

    out1 = paddle.unsqueeze(x, axis=0)
    print(out1.shape)  # [1, 5, 10]

    out2 = paddle.unsqueeze(x, axis=[0, 2]) 
    print(out2.shape)  # [1, 5, 1, 10]

    axis = paddle.to_tensor([0, 1, 2])
    out3 = paddle.unsqueeze(x, axis=axis) 
    print(out3.shape)  # [1, 1, 1, 5, 10]

    # 在动态图模式下，输出out1, out2, out3与输入x共享数据
    x[0, 0] = 10.
    print(out1[0, 0, 0]) # [10.]
    print(out2[0, 0, 0, 0]) # [10.]
    print(out3[0, 0, 0, 0, 0]) # [10.]
