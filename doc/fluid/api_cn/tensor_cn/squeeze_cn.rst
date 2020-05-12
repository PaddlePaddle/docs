.. _cn_api_paddle_tensor_squeeze
squeeze
-------------------------------

.. py:function:: paddle.tensor.squeeze(input, zxes, name=None)

该OP会根据axes压缩输入Tensor的维度。如果指定了axes，则会删除axes中指定的维度，axes指定的维度要等于1。如果没有指定axes，那么所有等于1的维度都会被删除。

**参数**：
        - **input** (Variable) - 输入任意维度的Tensor。 支持的数据类型：float32，float64，int8，int32，int64。
        - **axes** (list) - 输入一个或一列整数，代表要压缩的轴。axes的范围： [−rank(input),rank(input))] 。 axes为负数时， axes=axes+rank(input) 。
        - **name** (str，可选) - 一般无需设置，默认值为None。

**返回**：返回对维度进行压缩后的Tensor。数据类型与输入Tensor一致。

**返回类型**：Variable

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    with fluid.dygraph.guard():
        input_1 = np.random.random([5, 1, 10]).astype("int32")
        # input is a variable which shape is [5, 1, 10]
        input = fluid.dygraph.to_variable(input_1)

        output = paddle.fluid.layers.squeeze(input, axes=[1])
        # output.shape [5, 10]
