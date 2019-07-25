.. _cn_api_fluid_layers_square:

square
-------------------------------

.. py:function:: paddle.fluid.layers.square(x,name=None)

取平方激活函数。

.. math::
    out = x^2

参数:
    - **x** : 平方操作符的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：平方后的结果

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.square(data)











