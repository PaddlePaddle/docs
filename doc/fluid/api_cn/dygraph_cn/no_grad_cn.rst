.. _cn_api_fluid_dygraph_no_grad:

no_grad
-------------------------------

**注意：该API仅支持【动态图】模式**

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
            linear1 = fluid.Linear(1024, 4, bias_attr=False)
            linear2 = fluid.Linear(4, 4)
            ret = linear1(t)
            dy_ret = linear2(ret)

    test_layer()
