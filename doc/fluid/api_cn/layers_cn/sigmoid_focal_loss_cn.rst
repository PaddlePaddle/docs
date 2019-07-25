.. _cn_api_fluid_layers_sigmoid_focal_loss:

sigmoid_focal_loss
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid_focal_loss(x, label, fg_num, gamma=2, alpha=0.25)

**Sigmoid Focal loss损失计算**

focal损失用于解决在one-stage探测器的训练阶段存在的前景 - 背景类不平衡问题。 此运算符计算输入张量中每个元素的sigmoid值，然后计算focal损失。

focal损失计算过程：

.. math::

  loss_j = (-label_j * alpha * {(1 - \sigma(x_j))}^{gamma} * \log(\sigma(x_j)) -
  (1 - labels_j) * (1 - alpha) * {(\sigma(x_j)}^{ gamma} * \log(1 - \sigma(x_j)))
  / fg\_num, j = 1,...,K

其中，已知：

.. math::

  \sigma(x_j) = \frac{1}{1 + \exp(-x_j)}

参数：
    - **x**  (Variable) – 具有形状[N，D]的2-D张量，其中N是batch大小，D是类的数量（不包括背景）。 此输入是由前一个运算符计算出的logits张量。
    - **label**  (Variable) – 形状为[N，1]的二维张量，是所有可能的标签。
    - **fg_num**  (Variable) – 具有形状[1]的1-D张量，是前景的数量。
    - **gamma**  (float) –  用于平衡简单和复杂实例的超参数。 默认值设置为2.0。
    - **alpha**  (float) – 用于平衡正面和负面实例的超参数。 默认值设置为0.25。


返回：  具有形状[N，D]的2-D张量，即focal损失。

返回类型： out(Variable)

**代码示例**

..  code-block:: python


    import paddle.fluid as fluid

    input = fluid.layers.data(
        name='data', shape=[10,80], append_batch_size=False, dtype='float32')
    label = fluid.layers.data(
        name='label', shape=[10,1], append_batch_size=False, dtype='int32')
    fg_num = fluid.layers.data(
        name='fg_num', shape=[1], append_batch_size=False, dtype='int32')
    loss = fluid.layers.sigmoid_focal_loss(x=input,
                                           label=label,
                                           fg_num=fg_num,
                                           gamma=2.,
                                           alpha=0.25)




