.. _cn_api_fluid_layers_logsigmoid:

logsigmoid
-------------------------------

.. py:function:: paddle.fluid.layers.logsigmoid(x, name=None)

Logsigmoid激活函数。


.. math::

    out = \log \frac{1}{1 + e^{-x}}


参数:
    - **x** - LogSigmoid算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：        LogSigmoid算子的输出

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.logsigmoid(data)









