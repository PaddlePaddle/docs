.. _cn_api_fluid_LazyGuard:

LazyGuard
-------------------------------

.. py:class:: paddle.LazyGuard()



LazyGuard 是一个用于触发 `nn.Layer` 中参数延迟初始化的上下文管理器，在 `with LazyGuard():` 代码块下所有的 `nn.Layer` 在构造时，其内部的参数均不会立即申请内存空间。

输入范围是 `(-inf, inf)`，输出范围是 `[-1,1]`。

代码示例
::::::::::::

.. code-block:: python

    from paddle import LazyGuard
    from paddle.nn import Linear

    with LazyGuard():
        # w and b are initialized lazily and have no memory.
        net = Linear(10, 10)

    for param in net.parameters():
        # Initialize param and allocate memory explicitly.
        param.initialize()
