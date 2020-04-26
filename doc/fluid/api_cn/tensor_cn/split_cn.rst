.. _cn_api_paddle_tensor_split
split
-------------------------------

.. py:function:: paddle.tensor.split(input, num_or_sections, dim=-1, name=None)

该OP将输入Tensor分割成多个子Tensor。

**参数**：
       - **input** (Variable) - 输入变量，数据类型为float32，float64，int32，int64的多维Tensor或者LoDTensor。
       - **num_or_sections** (int|list|tuple) - 如果 num_or_sections 是一个整数，则表示Tensor平均划分为相同大小子Tensor的数量。如果 num_or_sections 是一个list或tuple，那么它的长度代表子Tensor的数量，它的元素可以是整数或者形状为[1]的Tensor，依次代表子Tensor需要分割成的维度的大小。list或tuple的长度不能超过输入Tensor待分割的维度的大小。在list或tuple中，至多有一个元素值为-1，表示该值是由input的维度和其他num_or_sections中元素推断出来的。例如对一个维度为[4,6,6]Tensor的第三维进行分割时，指定num_or_sections=[2,-1,1]，输出的三个Tensor维度分别为：[4,6,2]，[4,6,3]，[4,6,1]。
       - **dim** (int|Variable，可选) - 整数或者形状为[1]的Tensor，数据类型为int32或int64。表示需要分割的维度。如果dim < 0，则划分的维度为rank(input) + dim。默认值为-1。
       - **name** (str，可选) - 一般无需设置，默认值为None。

**返回**：分割后的Tensor列表。

**返回类型**：列表(Variable(Tensor|LoDTensor))，数据类型为int32，int64，float32，float64。

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    with fluid.dygraph.guard():
        input_1 = np.random.random([4, 6, 6]).astype("int32")
        # input is a variable which shape is [4, 6, 6]
        input = fluid.dygraph.to_variable(input_1)

        x0, x1, x2 = paddle.split(input, num_or_sections= 3, dim=1)
        # x0.shape [4, 2, 6]
        # x1.shape [4, 2, 6]
        # x2.shape [4, 2, 6]
