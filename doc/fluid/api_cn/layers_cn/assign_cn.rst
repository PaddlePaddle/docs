.. _cn_api_fluid_layers_assign:

assign
-------------------------------

.. py:function:: paddle.fluid.layers.assign(input,output=None)

该OP将输入Tensor或numpy数组拷贝至输出Tensor。

参数：
    - **input** (Variable|np.ndarray) - 输入Tensor或numpy数组，支持数据类型为float32, float64, int32和int64。
    - **output** (Variable，可选) - 输出Tensor。如果为None，则创建一个新的Tensor作为输出Tensor，默认值为None。

返回：输出Tensor，形状、数据类型、数据值和 ``input`` 一致。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    data = fluid.layers.fill_constant(shape=[3, 2], value=2.5, dtype='float64') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    result1 = fluid.layers.create_tensor(dtype='float64')
    fluid.layers.assign(data, result1) # result1 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    result2 = fluid.layers.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    result3 = fluid.layers.assign(np.array([[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]], dtype='float32')) # result3 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
