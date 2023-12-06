.. _cn_api_paddle_nn_functional_multi_margin_loss:

multi_margin_loss
-------------------------------

.. py:function:: paddle.nn.functional.multi_margin_loss(input, label, p:int = 1, margin: float = 1.0, weight=None, reduction: str = 'mean', name:str=None)

计算输入 `input` 和 `label` 间的多分类问题的 `hinge loss` 损失。


损失函数如果在没有的权重下计算每一个 mini-batch 的 loss 按照下列公式计算

.. math::
    \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, \text{margin} - input_i[label_i] + input_i[j])^p}{\text{C}}

其中 :math:`0 \leq j \leq \text{C}-1`, 且 :math:`j \neq label_i`， :math:`0 \leq i \leq \text{N}-1` N 为 batch 数量, C 为类别数量。

如果含有权重 `weight` 则损失函数按以下公式计算

.. math::
    \text{loss}(input_i, label_i) = \frac{\sum_{j} \max(0, weight[label_i] * (\text{margin} - input_i[label_i] + input_i[j]))^p}{\text{C}}

参数
:::::::::
    - **input** (Tensor) - :math:`[N, C]`，其中 N 是 batch_size， `C` 是类别数量。数据类型是 float32、float64。
    - **label** (Tensor) - :math:`[N, ]`。标签 ``label`` 的数据类型为 int32、int64。
    - **p** (int，可选) - 手动指定范数，默认为 1。
    - **margin** (float，可选) - 手动指定间距，默认为 1。
    - **weight** (Tensor，可选) - 权重值，默认为 None。如果给定则形状为 :math:`[C, ]`。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有：``'none'``， ``'mean'``， ``'sum'``。默认为 ``'mean'``，计算 Loss 的均值；设置为 ``'sum'`` 时，计算 Loss 的总和；设置为 ``'none'`` 时，则返回原始 Loss。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **input** (Tensor) - :math:`[N, C ]`，其中 N 是 batch_size，`C` 是类别问题。数据类型是 float32、float64。
    - **label** (Tensor) - :math:`[N, ]`，标签 ``label`` 的数据类型为 int32、int64。
    - **output** (Tensor) - 输出的 Tensor。如果 :attr:`reduction` 是 ``'none'``，则输出的维度为 :math:`[N, ]`，与输入 ``input`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``，则输出的维度为 :math:`[]` 。

返回
:::::::::
   返回计算的 Loss。

代码示例
:::::::::
COPY-FROM: paddle.nn.functional.multi_margin_loss
