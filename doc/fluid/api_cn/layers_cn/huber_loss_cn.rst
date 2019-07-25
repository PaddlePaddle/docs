.. _cn_api_fluid_layers_huber_loss:

huber_loss
-------------------------------

.. py:function:: paddle.fluid.layers.huber_loss(input, label, delta)

Huber损失是更具鲁棒性的损失函数。 huber损失可以评估输入对标签的合适度。 与MSE损失不同，Huber损失可更为稳健地处理异常值。

当输入和标签之间的距离大于delta时:

.. math::
        huber\_loss = delta * (label - input) - 0.5 * delta * delta

当输入和标签之间的距离小于delta时:

.. math::
        huber\_loss = 0.5 * (label - input) * (label - input)


参数:
  - **input** （Variable） - 此输入是前一个算子计算得到的概率。 第一个维度是批大小batch_size，最后一个维度是1。
  - **label** （Variable） - 第一个维度为批量大小batch_size且最后一个维度为1的真实值
  - **delta** （float） -  huber loss的参数，用于控制异常值的范围

返回： 形为[batch_size, 1]的huber loss.

返回类型:   huber_loss (Variable)



**代码示例**

..  code-block:: python

    import paddle.fluid as fluid

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    predict = fluid.layers.fc(input=x, size=1)
    label = fluid.layers.data(
        name='label', shape=[1], dtype='float32')
    loss = fluid.layers.huber_loss(
        input=predict, label=label, delta=1.0)





