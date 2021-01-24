.. _cn_paddle_nn_functional_loss_kl_div:

kl_div
-------------------------------

.. py:function:: paddle.nn.functional.kl_div(input, label, reduction='mean', name=None)

该算子计算输入(Input)和输入(Label)之间的Kullback-Leibler散度损失。注意其中输入(Input)应为对数概率值，输入(Label)应为概率值。

kL发散损失计算如下：

..  math::

    l(input, label) = label * (log(label) - input)


当 ``reduction``  为 ``none`` 时，输出损失与输入（x）形状相同，各点的损失单独计算，不会对结果做reduction 。

当 ``reduction``  为 ``mean`` 时，输出损失为[1]的形状，输出为所有损失的平均值。

当 ``reduction``  为 ``sum`` 时，输出损失为[1]的形状，输出为所有损失的总和。

当 ``reduction``  为 ``batchmean`` 时，输出损失为[N]的形状，N为批大小，输出为所有损失的总和除以批量大小。

参数
:::::::::
    - **input** (Tensor) - KL散度损失算子的输入张量。维度为[N, \*]的多维Tensor，其中N是批大小，\*表示任何数量的附加维度，数据类型为float32或float64。
    - **label** (Tensor) - KL散度损失算子的张量。与输入 ``input`` 的维度和数据类型一致的多维Tensor。
    - **reduction** (str，可选) - 要应用于输出的reduction类型，可用类型为‘none’ | ‘batchmean’ | ‘mean’ | ‘sum’，‘none’表示无reduction，‘batchmean’ 表示输出的总和除以批大小，‘mean’ 表示所有输出的平均值，‘sum’表示输出的总和。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置。默认值：None。
    
返回
:::::::::
Tensor KL散度损失。


代码示例
:::::::::

.. code-block:: python

    import paddle
    import numpy as np
    import paddle.nn.functional as F

    shape = (5, 20)
    input = np.random.uniform(-10, 10, shape).astype('float32')
    target = np.random.uniform(-10, 10, shape).astype('float32')

    # 'batchmean' reduction, loss shape will be [N]
    pred_loss = F.kl_div(paddle.to_tensor(input),
                            paddle.to_tensor(target), reduction='batchmean')
    # shape=[5]

    # 'mean' reduction, loss shape will be [1]
    pred_loss = F.kl_div(paddle.to_tensor(input),
                            paddle.to_tensor(target), reduction='mean')
    # shape=[1]

    # 'sum' reduction, loss shape will be [1]
    pred_loss = F.kl_div(paddle.to_tensor(input),
                            paddle.to_tensor(target), reduction='sum')
    # shape=[1]

    # 'none' reduction, loss shape is same with input shape
    pred_loss = F.kl_div(paddle.to_tensor(input),
                            paddle.to_tensor(target), reduction='none')
    # shape=[5, 20]

