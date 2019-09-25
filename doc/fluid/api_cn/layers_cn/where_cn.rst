.. _cn_api_fluid_layers_where:

where
-------------------------------

.. py:function:: paddle.fluid.layers.where(condition)

该OP计算输入元素中为True的元素在输入中的坐标（index）。
        
参数：
    - **condition** （Variable）– 输入秩至少为1的多维Tensor，数据类型是bool类型。

返回：输出condition元素为True的坐标（index），将所有的坐标（index）组成一个2-D的Tensor。

返回类型：Variable，数据类型是int64。
     
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


