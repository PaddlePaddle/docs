.. _cn_api_fluid_layers_kldiv_loss:

kldiv_loss
-------------------------------

.. py:function:: paddle.fluid.layers.kldiv_loss(x, target, reduction='mean', name=None)

该OP计算输入(X)和输入(Target)之间的Kullback-Leibler散度损失。注意其中输入(X)应为对数概率值，输入(Target)应为概率值。

kL发散损失计算如下：

..  math::

    l(x, y) = y * (log(y) - x)

:math:`x` 为输入（X），:math:`y` 输入（Target）。

当 ``reduction``  为 ``none`` 时，输出损失与输入（x）形状相同，各点的损失单独计算，不会对结果做reduction 。

当 ``reduction``  为 ``mean`` 时，输出损失为[1]的形状，输出为所有损失的平均值。

当 ``reduction``  为 ``sum`` 时，输出损失为[1]的形状，输出为所有损失的总和。

当 ``reduction``  为 ``batchmean`` 时，输出损失为[N]的形状，N为批大小，输出为所有损失的总和除以批量大小。

参数:
    - **x** (Variable) - KL散度损失算子的输入张量。维度为[N, \*]的多维Tensor，其中N是批大小，\*表示任何数量的附加维度，数据类型为float32或float64。
    - **target** (Variable) - KL散度损失算子的张量。与输入 ``x`` 的维度和数据类型一致的多维Tensor。
    - **reduction** (Variable)-要应用于输出的reduction类型，可用类型为‘none’ | ‘batchmean’ | ‘mean’ | ‘sum’，‘none’表示无reduction，‘batchmean’ 表示输出的总和除以批大小，‘mean’ 表示所有输出的平均值，‘sum’表示输出的总和。
    - **name** (None|str) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：Variable(Tensor) KL散度损失。

返回类型：变量(Variable)，数据类型与输入 ``x`` 一致。

**代码示例**：

.. code-block:: python

    # 'batchmean' reduction, loss shape 为[N]
    x = fluid.layers.data(name='x', shape=[4,2,2], dtype='float32') # shape=[-1, 4, 2, 2]
    target = fluid.layers.data(name='target', shape=[4,2,2], dtype='float32')
    loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='batchmean') # shape=[-1]

    # 'mean' reduction, loss shape 为[1]
    x = fluid.layers.data(name='x', shape=[4,2,2], dtype='float32') # shape=[-1, 4, 2, 2]
    target = fluid.layers.data(name='target', shape=[4,2,2], dtype='float32')
    loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='mean') # shape=[1]

    # 'sum' reduction, loss shape 为[1]
    x = fluid.layers.data(name='x', shape=[4,2,2], dtype='float32') # shape=[-1, 4, 2, 2]
    target = fluid.layers.data(name='target', shape=[4,2,2], dtype='float32')
    loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='sum') # shape=[1]

    # 'none' reduction, loss shape 与X相同
    x = fluid.layers.data(name='x', shape=[4,2,2], dtype='float32') # shape=[-1, 4, 2, 2]
    target = fluid.layers.data(name='target', shape=[4,2,2], dtype='float32')
    loss = fluid.layers.kldiv_loss(x=x, target=target, reduction='none') # shape=[-1, 4, 2, 2]







