.. _cn_api_fluid_layers_npair_loss:

npair_loss
-------------------------------

.. py:function:: paddle.fluid.layers.npair_loss(anchor, positive, labels, l2_reg=0.002)

**Npair Loss Layer**

参考阅读 `Improved Deep Metric Learning with Multi class N pair Loss Objective <http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf>`_

NPair损失需要成对的数据。NPair损失分为两部分：第一部分是嵌入向量上的L2正则化器；第二部分是以anchor的相似矩阵和正的相似矩阵为逻辑的交叉熵损失。

参数:
    - **anchor** (Variable) -  嵌入锚定图像的向量。尺寸=[batch_size, embedding_dims]
    - **positive** (Variable) -  嵌入正图像的向量。尺寸=[batch_size, embedding_dims]
    - **labels** (Variable) - 1维张量，尺寸=[batch_size]
    - **l2_reg** (float32) - 嵌入向量的L2正则化项，默认值：0.002

返回： npair loss，尺寸=[1]

返回类型：npair loss(Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    anchor = fluid.layers.data(
              name = 'anchor', shape = [18, 6], dtype = 'float32', append_batch_size=False)
    positive = fluid.layers.data(
              name = 'positive', shape = [18, 6], dtype = 'float32', append_batch_size=False)
    labels = fluid.layers.data(
              name = 'labels', shape = [18], dtype = 'float32', append_batch_size=False)

    npair_loss = fluid.layers.npair_loss(anchor, positive, labels, l2_reg = 0.002)






