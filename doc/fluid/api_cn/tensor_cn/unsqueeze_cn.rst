.. _cn_api_paddle_tensor_unsqueeze
unsqueeze
-------------------------------

.. py:function:: paddle.tensor.unsqueeze(input, axes, name=None)

该OP向输入（input）的shape中一个或多个位置（axes）插入维度。

**参数**：
        - **input** (Variable)- 多维 Tensor，数据类型为 float32， float64， int8， int32，或 int64。
        - **axes** (int|list|tuple|Variable) - 表示要插入维度的位置。数据类型是 int32 。如果 axes 的类型是 list 或 tuple，它的元素可以是整数或者形状为[1]的 Tensor 。如果 axes 的类型是 Variable，则是1-D Tensor。
        - **name** （str，可选）- 一般无需设置。默认值： None。

**返回**：扩展维度后的多维Tensor

**返回类型**：Variable

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    with fluid.dygraph.guard():
        input_1 = np.random.random([5, 10]).astype("int32")
        # input is a variable which shape is [5, 1, 10]
        input = fluid.dygraph.to_variable(input_1)

        output = paddle.unsqueeze(input, axes=[1])
        # output.shape [5, 1, 10]
