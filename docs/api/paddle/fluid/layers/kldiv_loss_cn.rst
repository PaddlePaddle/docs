.. _cn_api_fluid_layers_kldiv_loss:

kldiv_loss
-------------------------------

.. py:function:: paddle.fluid.layers.kldiv_loss(x, target, reduction='mean', name=None)




该 OP 计算输入(X)和输入(Target)之间的 Kullback-Leibler 散度损失。注意其中输入(X)应为对数概率值，输入(Target)应为概率值。

kL 发散损失计算如下：

..  math::

    l(x, y) = y * (log(y) - x)

:math:`x` 为输入（X），:math:`y` 输入（Target）。

当 ``reduction``  为 ``none`` 时，输出损失与输入（x）形状相同，各点的损失单独计算，不会对结果做 reduction 。

当 ``reduction``  为 ``mean`` 时，输出损失为[1]的形状，输出为所有损失的平均值。

当 ``reduction``  为 ``sum`` 时，输出损失为[1]的形状，输出为所有损失的总和。

当 ``reduction``  为 ``batchmean`` 时，输出损失为[N]的形状，N 为批大小，输出为所有损失的总和除以批量大小。

参数
::::::::::::

    - **x** (Variable) - KL 散度损失算子的输入 Tensor。维度为[N, \*]的多维 Tensor，其中 N 是批大小，\*表示任何数量的附加维度，数据类型为 float32 或 float64。
    - **target** (Variable) - KL 散度损失算子的 Tensor。与输入 ``x`` 的维度和数据类型一致的多维 Tensor。
    - **reduction** (Variable)-要应用于输出的 reduction 类型，可用类型为‘none’ | ‘batchmean’ | ‘mean’ | ‘sum’，‘none’表示无 reduction，‘batchmean’ 表示输出的总和除以批大小，‘mean’ 表示所有输出的平均值，‘sum’表示输出的总和。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
Variable(Tensor) KL 散度损失。

返回类型
::::::::::::
变量(Variable)，数据类型与输入 ``x`` 一致。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.kldiv_loss
