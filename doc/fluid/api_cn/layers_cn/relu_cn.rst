.. _cn_api_fluid_layers_relu:

relu
-------------------------------

.. py:function:: paddle.fluid.layers.relu(x, name=None)

ReLU（Rectified Linear Unit）激活函数

.. math::Out=max(0,x)


参数:
  - **x** (Variable) - 输入的多维 ``Tensor`` ，数据类型为：float32、float64。
  - **name** (str，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回: 与 ``x`` 维度相同数据类型相同的Tensor。

返回类型: Variable

**代码示例**:

..  code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name="x", shape=[3, 4], dtype="float32")
    output = fluid.layers.relu(x)
