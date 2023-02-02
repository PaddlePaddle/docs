.. _cn_api_incubate_identity_loss:

identity_loss
-------------------------------

.. py:function::  paddle.incubate.identity_loss(x, reduction='none')


用于在 IPU 动态图转静态图功能中标记网络的损失值，从而能够在 IPU 上为网络添加反向计算过程。算子以网络的损失值作为输入，并对输入做 reduction:

当 `reduction` 为 `none` 时，直接返回最原始的 `Out` 结果。

当 `reduction` 为 `mean` 时，最终的输出结果为：

.. math::
  Out = MEAN(Out)

当 `reduction` 为 `sum` 时，最终的输出结果为：

.. math::
  Out = SUM(Out)

参数
::::::::::::

    - **x** (Variable) - 输入 Tensor。维度为[N, \*]的多维 Tensor，其中 N 是批大小，\*表示任何数量的附加维度。数据类型在 CPU 上为 float32 或 float64，在 IPU 上为 float16 或 float32。
    - **reduction** (str|int，可选) - 指定应用于输出结果的计算方式，可选的 string 值有: ``'sum'``, ``'mean'``, ``'none'`` ，对应的 int 值分别为 0，1，2 。默认为 ``'none'``，直接返回输入 loss 的值。

返回
::::::::::::
Variable，根据 `reduction` 返回网络损失值的计算结果。

代码示例
::::::::::::

COPY-FROM: paddle.incubate.identity_loss
