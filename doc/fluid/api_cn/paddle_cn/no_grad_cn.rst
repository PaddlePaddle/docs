.. _cn_api_paddle_no_grad:

no_grad
-------------------------------

.. py:class:: paddle.fluid.dygraph.no_grad_

:api_attr: 命令式编程模式（动态图)


创建一个上下文来禁用动态图梯度计算。在此模式下，每次计算的结果都将具有stop_gradient=True。

也可以用作一个装饰器（需要创建实例对象作为装饰器）。

**代码示例**

..  code-block:: python

    import numpy as np
    import paddle

    paddle.disable_static()

    # 用作生成器

    data = np.array([[2, 3], [4, 5]]).astype('float32')
    l0 = paddle.nn.Linear(2, 2)  # l0.weight.gradient() is None
    l1 = paddle.nn.Linear(2, 2)
    with paddle.no_grad():
        # l1.weight.stop_gradient is False
        tmp = l1.weight * 2  # tmp.stop_gradient is True
    x = paddle.to_tensor(data)
    y = l0(x) + tmp
    o = l1(y)
    o.backward()
    print(tmp.gradient() is None)  # True
    print(l0.weight.gradient() is None)  # False

    # 用作装饰器
    @paddle.no_grad()
    def test_layer():
        inp = np.ones([3, 1024], dtype='float32')
        t = paddle.to_tensor(inp)
        linear1 = paddle.nn.Linear(1024, 4, bias_attr=False)
        linear2 = paddle.nn.Linear(4, 4)
        ret = linear1(t)
        dy_ret = linear2(ret)

    test_layer()
