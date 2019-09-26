.. _cn_api_fluid_layers_square:

square
-------------------------------

.. py:function:: paddle.fluid.layers.square(x,name=None)

该OP执行逐元素取平方运算。

.. math::
    out = x^2

参数:
    - **x** (Variable) - 任意维度的Tensor，支持的数据类型： float32，float64。
    - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None。

返回：返回取平方后的Tensor，维度和数据类型同输入一致。

返回类型：Variable

**代码示例**：

.. code-block:: python

        import paddle.fluid as fluid
        data = fluid.layers.data(name="input", shape=[32, 784], dtype='float32')
        result = fluid.layers.square(data) #result.shape=[32, 784], type=float32











