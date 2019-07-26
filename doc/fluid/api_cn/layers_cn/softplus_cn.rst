.. _cn_api_fluid_layers_softplus:

softplus
-------------------------------

.. py:function:: paddle.fluid.layers.softplus(x,name=None)

softplus激活函数。

.. math::
    out = \ln(1 + e^{x})

参数：
    - **x** - Softplus操作符的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：Softplus操作后的结果

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.softplus(data)











