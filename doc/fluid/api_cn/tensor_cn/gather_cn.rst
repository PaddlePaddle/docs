.. _cn_api_paddle_tensor_gather
gather
-------------------------------

.. py:function:: paddle.tensor.gather(input, index, overwrite=True)

根据索引 index 获取输入（input）的最外层维度的条目，并将它们拼接在一起。

.. math::

        Out=X[Index]

**参数**:
        - **input** (Variable) - 输入, 秩 ``rank >= 1`` , 支持的数据类型包括 int32、int64、float32、float64 和 uint8 (CPU)、float16（GPU） 。
        - **index** (Variable) - 索引，秩 ``rank = 1``, 数据类型为 int32 或 int64。
        - **overwrite** (bool) - 具有相同索引时在反向更新梯度的模式。如果为 ``True`` ，则使用覆盖模式更新相同索引的梯度；如果为 ``False`` ，则使用累积模式更新相同索引的梯度。默认值为 ``True`` 。

**返回**：和输入的秩相同的输出张量。

**返回类型**：Variable

**代码示例**：

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    with fluid.dygraph.guard():
        input_1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
        index_1 = np.array([0,1])
        input = fluid.dygraph.to_variable(input_1)
        index = fluid.dygraph.to_variable(index_1)
        output = paddle.fluid.layers.gather(input, index)
        # expected output: [[1, 2, 3],[4, 5, 6]]
