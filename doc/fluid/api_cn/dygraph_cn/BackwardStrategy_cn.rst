.. _cn_api_fluid_dygraph_BackwardStrategy:

BackwardStrategy
-------------------------------


.. py:class:: paddle.fluid.dygraph.BackwardStrategy

:api_attr: 命令式编程模式（动态图)



**注意：该API只在动态图下生效**

BackwardStrategy是描述动态图反向执行的策略，主要功能是定义动态图反向执行时的不同策略

**属性：**

.. py:attribute:: sort_sum_gradient

是否按照前向执行的逆序加和多个梯度，例如当 x_var（ :ref:`api_guide_Variable` ）作为多个OP（这里以 :ref:`cn_api_fluid_layers_scale` 为例）的输入时，其产生的梯度是否按照前向书写时的
逆序加和，默认为False


**代码示例**

.. code-block:: python

    import paddle
    import numpy as np
    import paddle.fluid as fluid
    
    x = np.ones([2, 2], np.float32)
    with paddle.imperative.guard():
        x_var = paddle.imperative.to_variable(x)
        sums_inputs = []
        # 这里x_var将作为多个输入scale的输入
        for _ in range(10):
            sums_inputs.append(paddle.scale(x_var))
        ret2 = paddle.sums(sums_inputs)
        loss2 = paddle.sum(ret2)
        backward_strategy = paddle.imperative.BackwardStrategy()
        backward_strategy.sort_sum_gradient = True
        loss2.backward(backward_strategy)

