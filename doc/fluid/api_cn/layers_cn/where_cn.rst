.. _cn_api_fluid_layers_where:

where
-------------------------------

.. py:function:: paddle.fluid.layers.where(condition)
     
返回一个秩为2的int64型张量，指定condition中真实元素的坐标。
     
输出的第一维是真实元素的数量，第二维是condition的秩（维数）。如果没有真实元素，则将生成空张量。
        
参数：
    - **condition** （Variable） - 秩至少为1的布尔型张量。

返回：存储一个二维张量的张量变量

返回类型：变量（Variable）
     
**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        import paddle.fluid.layers as layers
        import numpy as np
        # tensor 为 [True, False, True]
        condition = layers.assign(np.array([1, 0, 1], dtype='int32'))
        condition = layers.cast(condition, 'bool')
        out = layers.where(condition) # [[0], [2]]

        # tensor 为 [[True, False], [False, True]]
        condition = layers.assign(np.array([[1, 0], [0, 1]], dtype='int32'))
        condition = layers.cast(condition, 'bool')
        out = layers.where(condition) # [[0, 0], [1, 1]]

        # tensor 为 [False, False, False]
        condition = layers.assign(np.array([0, 0, 0], dtype='int32'))
        condition = layers.cast(condition, 'bool')
        out = layers.where(condition) # [[]]




