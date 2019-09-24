.. _cn_api_fluid_dygraph_PRelu:

PRelu
-------------------------------

.. py:class:: paddle.fluid.dygraph.PRelu(name_scope, mode, param_attr=None)

该OP实现了 ``PRelu`` 激活函数的三种激活方式。

计算公式如下：

.. math::
    y = max(0, x) + \alpha min(0, x)


参数：
    - **name_scope** (str) - 该类的名称。
    - **mode** (str) - 权重共享模式。共提供三种激活方式：

        .. code-block:: text
            
            all：所有元素使用同一个 :math:`[\alpha]` 值
            channel：在同一个通道中的元素使用同一个 :math:`[\alpha]` 值
            element：每一个元素有一个独立的 :math:`[\alpha]` 值

    - **param_attr** (ParamAttr, 可选) - 可学习权重 :math:`[\alpha]` 的参数属性。默认值：None。

返回：无

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    from paddle.fluid.dygraph.base import to_variable
    import numpy as np

    inp_np = np.ones([5, 200, 100, 100]).astype('float32')
    with fluid.dygraph.guard():
        inp_np = to_variable(inp_np)
        mode = 'channel'
        prelu = fluid.PRelu('prelu', mode=mode, param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(1.0)))
        dy_rlt = prelu(inp_np)


