.. _cn_api_fluid_dygraph_BackwardStrategy:

BackwardStrategy
-------------------------------


.. py:class:: paddle.fluid.dygraph.BackwardStrategy

:api_attr: 命令式编程模式（动态图)



**注意：该API只在动态图下生效**

BackwardStrategy是描述动态图反向执行时采用的不同策略，包括梯度聚合的顺序等。

**属性：**

.. py:attribute:: sort_sum_gradient

是否按照前向执行的逆序加和多个梯度，例如当 x_var（ :ref:`api_guide_Variable` ）作为多个OP（这里以 :ref:`cn_api_fluid_layers_scale` 为例）的输入时，其产生的梯度是否按照前向书写时的
逆序加和，默认为False。


**代码示例**

.. code-block:: python

    import numpy as np
    import paddle
    paddle.disable_static()
    x = np.ones([2, 2], np.float32)
    x_var = paddle.to_variable(x)
    sums_inputs = []
    # 这里x_var将作为多个输入scale的输入
    for _ in range(10):
        sums_inputs.append(paddle.scale(x_var))
    ret2 = paddle.sums(sums_inputs)
    loss2 = paddle.reduce_sum(ret2)
    backward_strategy = paddle.BackwardStrategy()
    backward_strategy.sort_sum_gradient = True
    loss2.backward(backward_strategy)






