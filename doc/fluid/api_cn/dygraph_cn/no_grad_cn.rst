.. _cn_api_fluid_dygraph_no_grad:

no_grad
-------------------------------

:api_attr: 命令式编程模式(动态图)

.. py:method:: paddle.fluid.dygraph.no_grad(func=None)


创建一个上下文来禁用动态图梯度计算。在此模式下，每次计算的结果都将具有stop_gradient=True。

也可以用作一个装饰器（确保不要用括号来初始化）。

代码示例
:::::::::

..  code-block:: python


    import numpy as np
    import paddle.fluid as fluid

    # 用作生成器
    data = np.array([[2, 3], [4, 5]]).astype('float32')
    with fluid.dygraph.guard():
        l0 = fluid.Linear(2, 2)  # l0.weight.gradient() is None
        l1 = fluid.Linear(2, 2)
        with fluid.dygraph.no_grad():
            # l1.weight.stop_gradient is False
            tmp = l1.weight * 2  # tmp.stop_gradient is True
        x = fluid.dygraph.to_variable(data)
        y = l0(x) + tmp
        o = l1(y)
        o.backward()
        print(tmp.gradient() is None)  # True
        print(l0.weight.gradient() is None)  # False
    
    # 用作装饰器
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
