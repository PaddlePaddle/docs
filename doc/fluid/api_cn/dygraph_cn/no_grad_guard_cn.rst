.. _cn_api_fluid_dygraph_no_grad_guard:

no_grad_guard
-------------------------------

**注意：该API仅支持【动态图】模式**

.. py:method:: paddle.fluid.dygraph.no_grad_guard()

创建一个上下文来禁用动态图梯度计算。在此模式下，每次计算的结果都将具有stop_gradient=True。

返回：无

**代码示例**

..  code-block:: python


    import numpy as np
    import paddle.fluid as fluid

    data = np.array([[2, 3], [4, 5]]).astype('float32')
    with fluid.dygraph.guard():
        l0 = fluid.Linear(2, 2)  # l0.weight._grad_ivar() is None
        l1 = fluid.Linear(2, 2)
        with fluid.dygraph.no_grad_guard():
            # l1.weight.stop_gradient is False
            tmp = l1.weight * 2  # tmp.stop_gradient is True
        x = fluid.dygraph.to_variable(data)
        y = l0(x) + tmp
        o = l1(y)
        o.backward()
        print(tmp._grad_ivar() is None)  # True
        print(l0.weight._grad_ivar() is None)  # False
