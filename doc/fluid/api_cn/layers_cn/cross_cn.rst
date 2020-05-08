.. _cn_api_tensor_linalg_cross:

cross
-------------------------------

.. py:function:: paddle.cross(input, other, dim=None)

该OP返回在 ``dim`` 维度上，两个张量 ``input`` 和 ``other`` 的向量积（叉积）。 ``input`` 和 ``other`` 必须有相同的形状，
且指定的 ``dim`` 维上 ``size`` 必须为3，如果 ``dim`` 未指定，默认选取第一个 ``size`` 等于3的维度。
        
**参数**：
    - **input** （Variable）– 第一个输入张量。
    - **other** （Variable）– 第二个输入张量。
    - **dim**    (int, optional) – 沿着此维进行叉积操作，若未指定，则默认选取第一个 ``size`` 等于3的维度

**返回**：
    - **Variable** ，数据类型同输入。
     
**代码示例**：

.. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import numpy as np

        data_x = np.array([[1.0, 1.0, 1.0],
                           [2.0, 2.0, 2.0],
                           [3.0, 3.0, 3.0]])
        data_y = np.array([[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]])

        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(data_x)
            y = fluid.dygraph.to_variable(data_y)
            out_z1 = paddle.cross(x, y)
            print(out_z1.numpy())
            #[[-1. -1. -1.]
            # [ 2.  2.  2.]
            # [-1. -1. -1.]]
            out_z2 = paddle.cross(x, y, dim=1)
            print(out_z2.numpy())
            #[[0. 0. 0.]
            # [0. 0. 0.]
            # [0. 0. 0.]]


