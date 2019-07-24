.. _cn_api_fluid_layers_scale:

scale
-------------------------------

.. py:function:: paddle.fluid.layers.scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None)

缩放算子

对输入张量应用缩放和偏移加法。

if ``bias_after_scale`` = True:

.. math::
                                Out=scale*X+bias

else:

.. math::
                                Out=scale*(X+bias)

参数:
        - **x** (Variable) - (Tensor) 要比例运算的输入张量（Tensor）。
        - **scale** (FLOAT) - 比例运算的比例因子。
        - **bias** (FLOAT) - 比例算子的偏差。
        - **bias_after_scale** (BOOLEAN) - 在缩放之后或之前添加bias。在某些情况下，对数值稳定性很有用。
        - **act** (basestring|None) - 应用于输出的激活函数。
        - **name** (basestring|None)- 输出的名称。

返回:        比例算子的输出张量(Tensor)

返回类型:        变量(Variable)

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
     
    x = fluid.layers.data(name="X", shape=[1, 2, 5, 5], dtype='float32')
    y = fluid.layers.scale(x, scale = 2.0, bias = 1.0)









