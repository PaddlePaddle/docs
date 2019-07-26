.. _cn_api_fluid_layers_softsign:

softsign
-------------------------------

.. py:function:: paddle.fluid.layers.softsign(x,name=None)


softsign激活函数。

.. math::
    out = \frac{x}{1 + |x|}

参数：
    - **x** : Softsign操作符的输入
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn


返回：Softsign操作后的结果

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.softsign(data)











