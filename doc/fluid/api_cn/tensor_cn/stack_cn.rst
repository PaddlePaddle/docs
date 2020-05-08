.. _cn_api_paddle_tensor_arange
stack
-------------------------------

.. py:function:: paddle.tensor.stack(x, axis=0)

该OP沿 axis 轴对输入 x 进行堆叠操作。

**参数**：
        - **x** (Variable|list(Variable)) – 输入 x 可以是单个Tensor，或是多个Tensor组成的列表。如果 x 是一个列表，那么这些Tensor的维度必须相同。 假设输入是N维Tensor [d0,d1,...,dn−1]，则输出变量的维度为N+1维 [d0,d1,...daxis−1,len(x),daxis...,dn−1] 。支持的数据类型: float32，float64，int32，int64。

        - **axis** (int, 可选) – 指定对输入Tensor进行堆叠运算的轴，有效 axis 的范围是: [−(R+1),R+1)]，R是输入中第一个Tensor的rank。如果 axis < 0，则 axis=axis+rank(x[0])+1 。axis默认值为0。

**返回**：堆叠运算后的Tensor，数据类型与输入Tensor相同。输出维度等于 rank(x[0])+1 维。

**返回类型**：Variable

**代码示例**:

.. code-block:: python
   
    import paddle
    import paddle.fluid as fluid
    import numpy as np
    data1 = np.array([[1.0, 2.0,3.0]])
    data2 = np.array([[3.0, 4.0, 5.0]])
    data3 = np.array([[5.0, 6.0,7.0]])
    with fluid.dygraph.guard():
        x1 = fluid.dygraph.to_variable(data1)
        x2 = fluid.dygraph.to_variable(data2)
        x3 = fluid.dygraph.to_variable(data3)
        result = paddle.stack([x1, x2, x3], axis=2)
        # result shape: [3, 1, 2]
        # result value: [[[1.0, 2.0]],
        #                [[3.0, 4.0]],
        #                [[5.0, 6.0]]]
