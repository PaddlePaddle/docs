.. _cn_api_paddle_nn_MultiLabelMarginLoss:

MultiLabelMarginLoss
-------------------------------

.. py:class:: paddle.nn.MultiLabelMarginLoss(reduction='mean', name=None)

创建一个 MultiLabelMarginLoss 的可调用类。通过计算输入 `input` 和 `label` 间的多类别多分类问题的 `hinge loss (margin-based loss)` 损失。

损失函数计算每一个 mini-batch 的 loss 按照下列公式计算

.. math::
    \text{loss}(input_i, label_i) = \frac{\sum_{j \in \text{valid_labels}} \sum_{k \neq \text{valid_labels}} \max(0, 1 - (input_i[\text{valid_labels}[j]] - input_i[k]))}{C}

其中 :math:`C` 是类别数量， :math:`\text{valid_labels}` 包含样本 :math:`i` 所有非负的标签索引（遇到第一个 -1 时停止），:math:`k` 遍历除了 :math:`\text{valid_labels}` 之外的所有类别索引。

该损失函数只考虑前面的非负标签值，允许不同样本具有不同数量的目标类别。

参数
:::::::::
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有： ``'none'`` 、 ``'mean'`` 、 ``'sum'`` 。默认为  ``'mean'`` ，计算 Loss 的均值；设置为  ``'sum'``  时，计算 Loss 的总和；设置为  ``'none'``  时，则返回原始 Loss。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

调用参数
:::::::::
    - **input** (Tensor) - 数据类型是 float32、float64。
    - **label** (Tensor) - 标签的数据类型为 int32、int64。标签值应该是类别索引（非负值）和 -1 值。-1 值会被忽略并停止处理每个样本。

形状
:::::::::
    - **input** (Tensor) - :math:`[N, C]`，其中 N 是 batch_size， C 是类别数量。
    - **label** (Tensor) - :math:`[N, C]`，与 input 形状相同。
    - **output** (Tensor) - 输出的 Tensor。如果 :attr:`reduction` 是  ``'none'`` ，则输出的维度为 :math:`[N]`。如果 :attr:`reduction` 是  ``'mean'``  或  ``'sum'`` ，则输出的维度为 :math:`[]` 。

返回
:::::::::
   返回计算 MultiLabelMarginLoss 的可调用对象。

代码示例
:::::::::
COPY-FROM: paddle.nn.MultiLabelMarginLoss
