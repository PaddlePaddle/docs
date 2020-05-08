.. _cn_api_tensor_manipulation_roll:

roll
-------------------------------

.. py:function:: paddle.roll(input, shifts, dims=None):

该OP沿着指定维度对输入 ``input`` 进行循环滚动，当元素移动到最后位置时，会从第一个位置重新插入。如果 ``dims`` 为 ``None`` ，则输入在被循环滚动之前，会先展平成 ``1-D Tensor`` ，滚动操作完成后恢复成原来的形状。

**参数**：
    - **input** （Variable）– 输入张量。
    - **shifts** (int|list|tuple) - 滚动位移。如果 ``shifts`` 是一个元组或者列表，则 ``dims`` 必须是相同大小的元组或者列表，输入张量将依次沿着每个维度滚动相应的数值。
    - **dim**    (int|list|tuple, optinal) – 滚动轴。

**返回**：
    - **Variable**，数据类型同输入。
     
**代码示例**：

.. code-block:: python

        import numpy as np
        import paddle
        import paddle.fluid as fluid

        data = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]])
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data)
            out_z1 = paddle.roll(x, shifts=1)
            print(out_z1.numpy())
            #[[9. 1. 2.]
            # [3. 4. 5.]
            # [6. 7. 8.]]
            out_z2 = paddle.roll(x, shifts=1, dims=0)
            print(out_z2.numpy())
            #[[7. 8. 9.]
            # [1. 2. 3.]
            # [4. 5. 6.]]


