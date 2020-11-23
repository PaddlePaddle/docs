.. _cn_api_paddle_tensor_squeeze
squeeze
-------------------------------

.. py:function:: paddle.tensor.squeeze(x, axis, name=None)

该OP会删除输入Tensor的Shape中尺寸为1的维度。如果指定了axis，则会删除axis中指定的尺寸为1的维度。如果没有指定axis，那么所有等于1的维度都会被删除。

.. code-block:: text

    Case 1:

        Input:
        x.shape = [1, 3, 1, 5]  # If axis is not provided, all dims equal of size 1 will be removed.
        axis = None
        Output:
        out.shape = [3, 5]

    Case 2:

        Input:
        x.shape = [1, 3, 1, 5]  # If axis is provided, it will remove the dimension(s) by given axis that of size 1.
        axis = 0
        Output:
        out.shape = [3, 1, 5]
    
    Case 3:

        Input:
        x.shape = [1, 3, 1, 5]  # If the dimension of one given axis (3) is not of size 1, the dimension remain unchanged. 
        axis = [0, 2, 3]
        Output:
        out.shape = [3, 5]

    Case 4:

        Input:
        x.shape = [1, 3, 1, 5]  # If axis is negative, axis = axis + ndim (number of dimensions in x). 
        axis = [-2]
        Output:
        out.shape = [1, 3, 5]

**参数**：
        - **x** (Tensor) - 输入的 `Tensor` ，数据类型为：float32、float64、bool、int8、int32、int64。
        - **axis** (int|list|tuple, 可选) - 输入一个或一列整数，代表要压缩的轴。axis的范围： [−ndim(x), ndim(x))] 。 如果axis为负数， 则axis=axis+ndim(x) 。
        - **name** (str, 可选) - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

**返回**：返回对维度进行压缩后的Tensor，数据类型与输入Tensor一致。

**返回类型**：Tensor

**代码示例**：

.. code-block:: python

    import paddle

    x = paddle.rand([5, 1, 10])
    output = paddle.squeeze(x, axis=1)
    print(output.shape)  # [5, 10]
