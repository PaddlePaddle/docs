.. _cn_api_fluid_layers_round:

round
-------------------------------

.. py:function:: paddle.fluid.layers.round(x, name=None)

Round取整激活函数。


.. math::
     out = [x]


参数:

    - **x** - round算子的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：        Round算子的输出。

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.round(data)



