.. _cn_api_fluid_layers_rsqrt:

rsqrt
-------------------------------

.. py:function:: paddle.fluid.layers.rsqrt(x, name=None)

rsqrt激活函数

请确保输入合法以免出现数字错误。

.. math::
    out = \frac{1}{\sqrt{x}}


参数:

    - **x** - rsqrt算子的输入 
    - **use_cudnn** (BOOLEAN) – （bool，默认为false）是否仅用于cudnn核，需要安装cudnn

返回：     rsqrt运算输出

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784])
        result = fluid.layers.rsqrt(data)



