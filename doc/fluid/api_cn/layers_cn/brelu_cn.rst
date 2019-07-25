.. _cn_api_fluid_layers_brelu:

brelu
-------------------------------

.. py:function:: paddle.fluid.layers.brelu(x, t_min=0.0, t_max=24.0, name=None)


BRelu 激活函数

.. math::   out=max(min(x,tmin),tmax)

参数:
    - **x** (Variable) - BReluoperator的输入
    - **t_min** (FLOAT|0.0) - BRelu的最小值
    - **t_max** (FLOAT|24.0) - BRelu的最大值
    - **name** (str|None) - 该层的名称(可选)。如果设置为None，该层将被自动命名


**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[2,3,16,16], dtype=”float32”)
    y = fluid.layers.brelu(x, t_min=1.0, t_max=20.0)






