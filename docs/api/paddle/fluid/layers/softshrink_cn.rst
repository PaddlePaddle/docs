.. _cn_api_fluid_layers_softshrink:

softshrink
-------------------------------

.. py:function:: paddle.fluid.layers.softshrink(x, alpha=None)




Softshrink 激活函数

.. math::
    out = \begin{cases}
        x - \alpha, \text{if } x > \alpha \\
        x + \alpha, \text{if } x < -\alpha \\
        0,  \text{otherwise}
        \end{cases}

参数
::::::::::::

    - **x** (Variable0 - 张量（Tensor）
    - **alpha** (float) - 上面公式中 alpha 的值

返回
::::::::::::
 张量(Tensor)

返回类型
::::::::::::
 变量(Variable)

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.softshrink
