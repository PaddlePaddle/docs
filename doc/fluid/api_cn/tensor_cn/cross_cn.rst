.. _cn_api_tensor_linalg_cross:

cross
-------------------------------

.. py:function:: paddle.cross(x, y, axis=None, name=None)

:alias_main: paddle.cross
:alias: paddle.cross,paddle.tensor.cross,paddle.tensor.linalg.cross



计算张量 ``x`` 和 ``y`` 在 ``axis`` 维度上的向量积（叉积）。 ``x`` 和 ``y`` 必须有相同的形状，
且指定的 ``axis`` 的长度必须为3. 如果未指定 ``axis`` ，默认选取第一个长度为3的 ``axis`` .
        
**参数**：
    - **x** （Variable）– 第一个输入张量。
    - **y** （Variable）– 第二个输入张量。
    - **axis**  (int, optional) – 沿着此维进行向量积操作。默认选取第一个长度为3的 ``axis`` .
    - **name** （str，可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

**返回**：向量积的结果。

**返回类型**：Variable

**代码示例**：

.. code-block:: python

        import paddle
        from paddle.imperative import to_variable
        import numpy as np

        paddle.enable_imperative()
        
        data_x = np.array([[1.0, 1.0, 1.0],
                           [2.0, 2.0, 2.0],
                           [3.0, 3.0, 3.0]])
        data_y = np.array([[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]])
        x = to_variable(data_x)
        y = to_variable(data_y)

        z1 = paddle.cross(x, y)
        print(z1.numpy())
        # [[-1. -1. -1.]
        #  [ 2.  2.  2.]
        #  [-1. -1. -1.]]

        z2 = paddle.cross(x, y, axis=1)
        print(z2.numpy())
        # [[0. 0. 0.]
        #  [0. 0. 0.]
        #  [0. 0. 0.]]


