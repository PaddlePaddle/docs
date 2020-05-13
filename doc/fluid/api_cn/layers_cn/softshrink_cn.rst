.. _cn_api_fluid_layers_softshrink:

softshrink
-------------------------------

.. py:function:: paddle.fluid.layers.softshrink(x, alpha=None)

:alias_main: paddle.nn.functional.softshrink
:alias: paddle.nn.functional.softshrink,paddle.nn.functional.activation.softshrink
:old_api: paddle.fluid.layers.softshrink



Softshrink激活函数

.. math::
    out = \begin{cases}
        x - \alpha, \text{if } x > \alpha \\
        x + \alpha, \text{if } x < -\alpha \\
        0,  \text{otherwise}
        \end{cases}

参数：
    - **x** (Variable0 - 张量（Tensor）
    - **alpha** (float) - 上面公式中alpha的值

返回: 张量(Tensor)

返回类型: 变量(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name="input", shape=[32, 784])
    result = fluid.layers.softshrink(data)












