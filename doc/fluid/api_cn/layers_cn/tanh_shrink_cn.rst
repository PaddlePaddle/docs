.. _cn_api_fluid_layers_tanh_shrink:

tanh_shrink
-------------------------------

.. py:function:: paddle.fluid.layers.tanh_shrink(x, name=None)

tanh_shrink激活函数。

.. math::
    out = x - \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}

参数:

    - **x** - TanhShrink算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：     tanh_shrink算子的输出

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.tanh_shrink(data)









