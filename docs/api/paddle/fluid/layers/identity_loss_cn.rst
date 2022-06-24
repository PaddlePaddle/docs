.. _cn_api_fluid_layers_identity_loss:

identity_loss
-------------------------------

.. py:function::  paddle.fluid.layers.identity_loss(x, reduction='none')


用于在IPU动态图转静态图功能中标记网络的损失值，从而能够在IPU上为网络添加反向计算过程。该OP以网络的损失值作为输入，并对输入做reduction:

当 `reduction` 为 `none` 时，直接返回最原始的 `Out` 结果。

当 `reduction` 为 `mean` 时，最终的输出结果为：

.. math::
  Out = MEAN(Out)

当 `reduction` 为 `sum` 时，最终的输出结果为：

.. math::
  Out = SUM(Out)

参数
::::::::::::

    - **x** (Variable) - 输入张量。维度为[N, \*]的多维Tensor，其中N是批大小，\*表示任何数量的附加维度。数据类型在CPU上为float32或float64，在IPU上为float16或float32。
    - **reduction** (str|int，可选) - 指定应用于输出结果的计算方式，可选的string值有: ``'mean'``, ``'sum'``, ``'none'`` ，对应的int值分别为0，1，2 。默认为 ``'none'``，直接返回输入loss的值；设置为 ``'mean'`` 时，返回输入loss的均值；设置为 ``'sum'`` 时，计算输入loss的总和。

返回
::::::::::::
Variable，根据 `reduction` 返回网络损失值的计算结果。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.identity_loss
