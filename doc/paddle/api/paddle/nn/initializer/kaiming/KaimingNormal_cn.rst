.. _cn_api_nn_initializer_KaimingNormal:

KaimingNormal
-------------------------------

.. py:class:: paddle.nn.initializer.KaimingNormal(fan_in=None)




该接口实现Kaiming正态分布方式的权重初始化

该接口为权重初始化函数，方法来自Kaiming He，Xiangyu Zhang，Shaoqing Ren 和 Jian Sun所写的论文: `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/abs/1502.01852>`_ 。这是一个鲁棒性特别强的初始化方法，并且适应了非线性激活函数（rectifier nonlinearities）。

在正态分布中，均值为0，标准差为：

.. math::

    \sqrt{\frac{2.0}{fan\_in}}

参数：
    - **fan_in** (float16|float32) - Kaiming Normal Initializer的fan_in。如果为None，fan_in沿伸自变量，多设置为None

返回：对象

.. note:: 

    在大多数情况下推荐设置fan_in为None

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.nn as nn
    linear = nn.Linear(2, 4, weight_attr=nn.initializer.KaimingNormal())
    data = paddle.rand([30, 10, 2], dtype='float32')
    res = linear(data)
