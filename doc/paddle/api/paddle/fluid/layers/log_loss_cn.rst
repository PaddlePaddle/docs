.. _cn_api_fluid_layers_log_loss:

log_loss
-------------------------------

.. py:function:: paddle.fluid.layers.log_loss(input, label, epsilon=0.0001, name=None)




**负log loss层**

该 OP 对输入的预测结果和目标标签进行计算，返回负对数损失值。

.. math::

    Out = -label * \log{(input + \epsilon)} - (1 - label) * \log{(1 - input + \epsilon)}


参数:
  - **input** (Variable) – 形为 [N x 1] 的二维张量, 其中 N 为 batch 大小。该输入是由前驱算子计算得来的概率， 数据类型是 float32。
  - **label** (Variable) – 形为 [N x 1] 的二维张量，真值标签， 其中 N 为 batch 大小，数据类型是 float32。
  - **epsilon** (float) – epsilon
  - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： 形为[N x 1]的二维张量，计算出的负log_loss值，数据类型为 float32

返回类型: Variable


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    label = fluid.data(name='label', shape=[None, 1], dtype='float32')
    prob = fluid.data(name='prob', shape=[None, 1], dtype='float32')
    cost = fluid.layers.log_loss(input=prob, label=label)











