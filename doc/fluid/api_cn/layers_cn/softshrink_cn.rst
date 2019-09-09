.. _cn_api_fluid_layers_softshrink:

softshrink
-------------------------------

.. py:function:: paddle.fluid.layers.softshrink(x, alpha=None)

Softshrink激活算子

.. math::
        out = \begin{cases}
                    x - lpha, ext{if } x > lpha \
                    x + lpha, ext{if } x < lpha \
                    0,  \text{otherwise}
              \end{cases}

参数：
        - **x** - Softshrink算子的输入
        - **alpha** （FLOAT）- 非负偏移量。

返回：       Softshrink算子的输出

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.softshrink(x=data, alpha=0.3)












