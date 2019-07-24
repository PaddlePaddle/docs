.. _cn_api_fluid_layers_log_loss:

log_loss
-------------------------------

.. py:function:: paddle.fluid.layers.log_loss(input, label, epsilon=0.0001, name=None)

**负log loss层**

该层对输入的预测结果和目的标签进行计算，返回负log loss损失值。

.. math::

    Out = -label * \log{(input + \epsilon)} - (1 - label) * \log{(1 - input + \epsilon)}


参数:
  - **input** (Variable|list) – 形为[N x 1]的二维张量, 其中N为batch大小。 该输入是由先前运算得来的概率集。
  - **label** (Variable|list) – 形为[N x 1]的二维张量，承载着正确标记的数据， 其中N为batch大小。
  - **epsilon** (float) – epsilon
  - **name** (string) – log_loss层的名称

返回： 形为[N x 1]的二维张量，承载着负log_loss值

返回类型: 变量（Variable）


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    prob = fluid.layers.data(name='prob', shape=[10], dtype='float32')
    cost = fluid.layers.log_loss(input=prob, label=label)











