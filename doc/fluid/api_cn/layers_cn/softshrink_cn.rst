.. _cn_api_fluid_layers_softshrink:

softshrink
-------------------------------

.. py:function:: paddle.fluid.layers.softshrink(x, name=None)

Softshrink激活算子

.. math::
        out = \begin{cases}
                    x - \lambda, \text{if } x > \lambda \\
                    x + \lambda, \text{if } x < -\lambda \\
                    0,  \text{otherwise}
              \end{cases}

参数：
        - **x** - Softshrink算子的输入
        - **lambda** （FLOAT）- 非负偏移量。

返回：       Softshrink算子的输出

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.softshrink(data)












