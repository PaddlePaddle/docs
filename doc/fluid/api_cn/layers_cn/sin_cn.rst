.. _cn_api_fluid_layers_sin:

sin
-------------------------------

.. py:function:: paddle.fluid.layers.sin(x, name=None)

正弦sine激活函数。

.. math::
     out = sin(x)


参数:

    - **x** - sin算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn


返回：        Sin算子的输出。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.sin(data)













