.. _cn_api_fluid_layers_cos:

cos
-------------------------------

.. py:function:: paddle.fluid.layers.cos(x, name=None)

Cosine余弦激活函数。

.. math::

    out = cos(x)



参数:

    - **x** - cos算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn


返回：        Cos算子的输出

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.cos(data)








