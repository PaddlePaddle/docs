.. _cn_api_fluid_layers_exp:

exp
-------------------------------

.. py:function:: paddle.fluid.layers.exp(x, name=None)

Exp激活函数(Exp指以自然常数e为底的指数运算)。

.. math::
    out = e^x

参数:

    - **x** - Exp算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn


返回：       Exp算子的输出

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.exp(data)









