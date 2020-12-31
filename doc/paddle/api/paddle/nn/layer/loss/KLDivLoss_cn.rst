KLDivLoss
-------------------------------

.. py:class:: paddle.nn.KLDivLoss(reduction='mean')

该算子计算输入(Input)和输入(Label)之间的Kullback-Leibler散度损失。注意其中输入(Input)应为对数概率值，输入(Label)应为概率值。

kL发散损失计算如下：

..  math::

    l(input, label) = label * (log(label) - input)


当 ``reduction``  为 ``none`` 时，输出损失与输入（input）形状相同，各点的损失单独计算，不会对结果做reduction 。

当 ``reduction``  为 ``mean`` 时，输出损失为[1]的形状，输出为所有损失的平均值。

当 ``reduction``  为 ``sum`` 时，输出损失为[1]的形状，输出为所有损失的总和。

当 ``reduction``  为 ``batchmean`` 时，输出损失为[N]的形状，N为批大小，输出为所有损失的总和除以批量大小。

参数:
    - **reduction** (str，可选) - 要应用于输出的reduction类型，可用类型为‘none’ | ‘batchmean’ | ‘mean’ | ‘sum’，‘none’表示无reduction，‘batchmean’ 表示输出的总和除以批大小，‘mean’ 表示所有输出的平均值，‘sum’表示输出的总和。
    
形状:
    - **input** (Tensor): - 输入的Tensor，维度是[N, *], 其中N是batch size， `*` 是任意数量的额外维度。数据类型为：float32、float64。
    - **label** (Tensor): - 标签，维度是[N, *], 与 ``input`` 相同。数据类型为：float32、float64。
    - **output** (Tensor): - 输入 ``input`` 和标签 ``label`` 间的kl散度。如果 `reduction` 是 ``'none'``, 则输出Loss的维度为 [N, *], 与输入 ``input`` 相同。如果 `reduction` 是 ``'batchmean'`` 、 ``'mean'`` 或 ``'sum'``, 则输出Loss的维度为 [1]。

**代码示例：**

.. code-block:: python

    import paddle
    import numpy as np
    import paddle.nn as nn

    shape = (5, 20)
    x = np.random.uniform(-10, 10, shape).astype('float32')
    target = np.random.uniform(-10, 10, shape).astype('float32')

    # 'batchmean' reduction, loss shape will be [N]
    kldiv_criterion = nn.KLDivLoss(reduction='batchmean')
    pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                paddle.to_tensor(target))
    # shape=[5]

    # 'mean' reduction, loss shape will be [1]
    kldiv_criterion = nn.KLDivLoss(reduction='mean')
    pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                paddle.to_tensor(target))
    # shape=[1]

    # 'sum' reduction, loss shape will be [1]
    kldiv_criterion = nn.KLDivLoss(reduction='sum')
    pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                paddle.to_tensor(target))
    # shape=[1]

    # 'none' reduction, loss shape is same with X shape
    kldiv_criterion = nn.KLDivLoss(reduction='none')
    pred_loss = kldiv_criterion(paddle.to_tensor(x),
                                paddle.to_tensor(target))
    # shape=[5, 20]

