.. _cn_api_fluid_dygraph_BackwardStrategy:

BackwardStrategy
-------------------------------

.. py:class:: paddle.fluid.dygraph.BackwardStrategy

BackwardStrategy是描述反向过程的描述符，现有如下功能:

1. ``sort_sum_gradient`` 按回溯逆序将梯度加和


**代码示例**

.. code-block:: python

    import numpy as np
    import paddle.fluid as fluid
    from paddle.fluid import FC

    x = np.ones([2, 2], np.float32)
    with fluid.dygraph.guard():
        inputs2 = []
        for _ in range(10):
            inputs2.append(fluid.dygraph.base.to_variable(x))
        ret2 = fluid.layers.sums(inputs2)
        loss2 = fluid.layers.reduce_sum(ret2)
        backward_strategy = fluid.dygraph.BackwardStrategy()
        backward_strategy.sort_sum_gradient = True
        loss2.backward(backward_strategy)




