.. _cn_api_fluid_dygraph_declarative:

declarative
-------------------------------

.. py:decorator:: paddle.fluid.dygraph.jit.declarative

本装饰器将函数内的动态图API转化为静态图API。此装饰器自动处理静态图模式下的Program和Executor，并将结果作为动态图Tensor返回。输出的动态图Tensor可以继续进行动态图训练、预测或其他运算。如果被装饰的函数里面调用其他动态图函数，被调用的函数也会被转化为静态图函数。

**示例代码**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    from paddle.fluid.dygraph.jit import declarative

    fluid.enable_dygraph()

    @declarative
    def func(x):
        x = fluid.dygraph.to_variable(x)
        if fluid.layers.mean(x) < 0:
            x_v = x - 1
        else:
            x_v = x + 1
        return x_v

    x = np.ones([1, 2])
    x_v = func(x)
    print(x_v.numpy()) # [[2. 2.]]

