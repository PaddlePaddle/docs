.. _cn_api_fluid_dygraph_no_grad:

no_grad
-------------------------------

.. py:method:: paddle.fluid.dygraph.no_grad(func)

在动态图模式中，此装饰器将会避免 ``func`` 被装饰时创建反向传播网络。

参数:
    - **func** (str) – 不需要梯度的函数。

**代码示例**

..  code-block:: python


    import numpy as np
    import paddle.fluid as fluid

    @fluid.dygraph.no_grad
    def test_layer():
        with fluid.dygraph.guard():
            inp = np.ones([3, 1024], dtype='float32')
            t = fluid.dygraph.base.to_variable(inp)
            fc1 = fluid.Linear(1024, 4, bias_attr=False)
            fc2 = fluid.Linear(4, 4)
            ret = fc1(t)
            dy_ret = fc2(ret)

    test_layer()
