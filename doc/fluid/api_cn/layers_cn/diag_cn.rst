.. _cn_api_fluid_layers_diag:

diag
-------------------------------

.. py:function:: paddle.fluid.layers.diag(diagonal)

该OP创建一个方阵，使用输入diagonal来指定方阵的对角线元素的值。

参数：
    - **diagonal** (Variable|numpy.ndarray) — 数据shape为 :math:`[N]` 一维Tensor，会把该Tensor的元素赋在方阵的对角线上。数据类型可以是 float32，float64，int32，int64。

返回：存储着方阵的Tensor，对角线值是输入Tensor diagonal的值， 数据shape为 :math:`[N, N]` 二维Tensor。

返回类型：Variable，数据类型和输入数据类型一致。

**代码示例**：

.. code-block:: python

        #  [3, 0, 0]
        #  [0, 4, 0]
        #  [0, 0, 5]

        import paddle.fluid as fluid
        import numpy as np
        diagonal = np.arange(3, 6, dtype='int32')
        data = fluid.layers.diag(diagonal)
        # diagonal.shape=(3,) data.shape=(3, 3)




