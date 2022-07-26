.. _cn_api_fluid_layers_hard_shrink:

hard_shrink
-------------------------------

.. py:function:: paddle.fluid.layers.hard_shrink(x,threshold=None)




HardShrink激活函数(HardShrink activation operator)


.. math::

  out = \begin{cases}
        x, \text{if } x > \lambda \\
        x, \text{if } x < -\lambda \\
        0,  \text{otherwise}
      \end{cases}

参数
::::::::::::

    - **x** - HardShrink激活函数的输入
    - **threshold** (FLOAT)-HardShrink激活函数的threshold值。[默认：0.5]

返回
::::::::::::
HardShrink激活函数的输出

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.hard_shrink