.. _cn_api_fluid_layers_kldiv_loss:

kldiv_loss
-------------------------------

.. py:function:: paddle.fluid.layers.kldiv_loss(x, target, reduction='mean', name=None)

此运算符计算输入（x）和输入（Target）之间的Kullback-Leibler发散损失。

kL发散损失计算如下：

..  math::

    l(x, y) = y * (log(y) - x)

:math:`x` 为输入（x），:math:`y` 输入（Target）。

当 ``reduction``  为 ``none`` 时，输出损失与输入（x）形状相同，各点的损失单独计算，不应用reduction 。

当 ``reduction``  为 ``mean`` 时，输出损失为[1]的形状，损失值为所有损失的平均值。

当 ``reduction``  为 ``sum`` 时，输出损失为[1]的形状，损失值为所有损失的总和。

当 ``reduction``  为 ``batchmean`` 时，输出损失为[1]的形状，损失值为所有损失的总和除以批量大小。

参数:
    - **x** (Variable) - KL发散损失算子的输入张量。这是一个形状为[N, \*]的张量，其中N是批大小，\*表示任何数量的附加维度
    - **target** (Variable) - KL发散损失算子的张量。这是一个具有输入（x）形状的张量
    - **reduction** (Variable)-要应用于输出的reduction类型，可用类型为‘none’ | ‘batchmean’ | ‘mean’ | ‘sum’，‘none’表示无reduction，‘batchmean’ 表示输出的总和除以批大小，‘mean’ 表示所有输出的平均值，‘sum’表示输出的总和。
    - **name** (str, default None) - 该层的名称

返回：KL发散损失

返回类型：kldiv_loss (Variable)

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[4,2,2], dtype='float32')
    target = fluid.layers.data(name='target', shape=[4,2,2], dtype='float32')
    loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='batchmean')







