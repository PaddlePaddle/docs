.. _cn_api_fluid_layers_sqrt:

sqrt
-------------------------------

.. py:function:: paddle.fluid.layers.sqrt(x, name=None)

算数平方根激活函数。

请确保输入是非负数。有些训练当中，会出现输入为接近零的负值，此时应加上一个小值epsilon（1e-12）将其变为正数从而正确运算并进行后续的操作。


.. math::
    out = \sqrt{x}

参数:

    - **x** - Sqrt算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：       Sqrt算子的输出。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.sqrt(data)













