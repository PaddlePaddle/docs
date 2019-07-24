.. _cn_api_fluid_layers_tanh:

tanh
-------------------------------

.. py:function:: paddle.fluid.layers.tanh(x, name=None)




tanh 激活函数。

.. math::
    out = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}


参数:

    - **x** - Tanh算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：     Tanh算子的输出。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.tanh(data)













