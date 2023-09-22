.. _cn_api_paddle_nn_functional_multi_label_soft_margin_loss:

multi_label_soft_margin_loss
-------------------------------

.. py:function:: paddle.nn.functional.multi_label_soft_margin_loss(input, label, weight=None, reduction='mean', name=None)

计算输入 `input` 和 `label` 间的 `margin-based loss` 损失。


损失函数按照下列公式计算

.. math::
    \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.shape}(0)}

如果添加权重则再乘以对应的权重值


最后，添加 `reduce` 操作到前面的输出 Out 上。当 `reduction` 为 `none` 时，直接返回最原始的 `Out` 结果。当 `reduction` 为 `mean` 时，返回输出的均值 :math:`Out = MEAN(Out)` 。当 `reduction` 为 `sum` 时，返回输出的求和 :math:`Out = SUM(Out)` 。


参数
:::::::::
    - **input** (Tensor) - :math:`[N, *]` , 其中 N 是 batch_size， `*` 是任意其他维度。数据类型是 float32、float64。
    - **label** (Tensor) - :math:`[N, *]` , 标签 ``label`` 的维度、数据类型与输入 ``input`` 相同。
    - **weight** (Tensor，可选) - 手动设定权重，默认为 None
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有: ``'none'``， ``'mean'``， ``'sum'`` 。默认为 ``'mean'``，计算 Loss 的均值；设置为 ``'sum'`` 时，计算 Loss 的总和；设置为 ``'none'`` 时，则返回原始 Loss。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **input** (Tensor) - :math:`[N, *]` , 其中 N 是 batch_size， `*` 是任意其他维度。数据类型是 float32、float64。
    - **label** (Tensor) - :math:`[N, *]` ，标签 ``label`` 的维度、数据类型与输入 ``input`` 相同。
    - **output** (Tensor) - 输出的 Tensor。如果 :attr:`reduction` 是 ``'none'``, 则输出的维度为 :math:`[N, *]` , 与输入 ``input`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``， 则输出的维度为 :math:`[]` 。

返回
:::::::::
    输出的 Tensor。如果 :attr:`reduction` 是 ``'none'``, 则输出的维度为 :math:`[N, *]` ， 与输入 ``input`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 `'sum'``， 则输出的维度为 :math:`[]` 。


代码示例
:::::::::
COPY-FROM: paddle.nn.functional.multi_label_soft_margin_loss
