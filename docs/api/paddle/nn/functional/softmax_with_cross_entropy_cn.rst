.. _cn_api_fluid_layers_softmax_with_cross_entropy:

softmax_with_cross_entropy
-------------------------------

.. py:function:: paddle.nn.functional.softmax_with_cross_entropy(logits, label, soft_label=False, ignore_index=-100, numeric_stable_mode=True, return_softmax=False, axis=-1)


实现了 softmax 交叉熵损失函数。该函数会将 softmax 操作、交叉熵损失函数的计算过程进行合并，从而提供了数值上更稳定的梯度值。

因为该运算对 ``logits`` 的 ``axis`` 维执行 softmax 运算，所以它需要未缩放的 ``logits``。该运算不应该对 softmax 运算的输出进行操作，否则会产生错误的结果。

当 ``soft_label`` 为 ``False`` 时，``label`` 除了 ``axis`` 维度上的形状为 1，其余维度和 ``logits`` 一致，表示一批数据中的每一个样本仅可分类到一个类别。

涉及到的等式如下：

1. 硬标签（每个样本仅可分到一个类别）

.. math::
   loss_j =  -\text{logits}_{label_j} +\log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right), j = 1,..., K

2. 软标签（每个样本以一定的概率被分配至多个类别中，概率和为 1）

.. math::
   loss_j =  -\sum_{i=0}^{K}\text{label}_i\left(\text{logits}_i - \log\left(\sum_{i=0}^{K}\exp(\text{logits}_i)\right)\right), j = 1,...,K

3. 如果 ``numeric_stable_mode`` 为 ``True`` ，softmax 结果首先经由下式计算得出，然后使用 softmax 结果和 ``label`` 计算交叉熵损失。

.. math::
    max_j           &= \max_{i=0}^{K}{\text{logits}_i} \\
    log\_max\_sum_j &= \log\sum_{i=0}^{K}\exp(logits_i - max_j)\\
    softmax_j &= \exp(logits_j - max_j - {log\_max\_sum}_j)

参数
::::::::::::

  - **logits** (Tensor) - 维度为任意维的多维 ``Tensor``，数据类型为 float32 或 float64。表示未缩放的输入。
  - **label** (Tensor) - 如果 ``soft_label`` 为 True， ``label`` 是一个和 ``logits`` 维度相同的的 ``Tensor``。如果 ``soft_label`` 为 False， ``label`` 是一个在 axis 维度上大小为 1，其它维度上与 ``logits`` 维度相同的 ``Tensor`` 。
  - **soft_label** (bool，可选) - 指明是否将输入标签当作软标签。默认值：False。
  - **ignore_index** (int，可选) - 指明要无视的目标值，使其不对输入梯度有贡献。仅在 ``soft_label`` 为 False 时有效，默认值：kIgnoreIndex（-100）。
  - **numeric_stable_mode** (bool，可选) – 指明是否使用一个具有更佳数学稳定性的算法。仅在 ``soft_label`` 为 False 的 GPU 模式下生效。若 ``soft_label`` 为 True 或者执行设备为 CPU，算法一直具有数学稳定性。注意使用稳定算法时速度可能会变慢。默认值：True。
  - **return_softmax** (bool，可选) – 指明是否在返回交叉熵计算结果的同时返回 softmax 结果。默认值：False。
  - **axis** (int，可选) – 执行 softmax 计算的维度索引。其范围为 :math:`[-1，rank-1]`，其中 ``rank`` 是输入 ``logits`` 的秩。默认值：-1。

返回
::::::::::::

  - 如果 ``return_softmax`` 为 False，则返回交叉熵损失结果的 ``Tensor``，数据类型和 ``logits`` 一致，除了 ``axis`` 维度上的形状为 1，其余维度和 ``logits`` 一致。
  - 如果 ``return_softmax`` 为 True，则返回交叉熵损失结果的 ``Tensor`` 和 softmax 结果的 ``Tensor`` 组成的元组。其中交叉熵损失结果的数据类型和 ``logits`` 一致，除了 ``axis`` 维度上的形状为 1，其余维度上交叉熵损失结果和 ``logits`` 一致；softmax 结果的数据类型和 ``logits`` 一致，维度和 ``logits`` 一致。


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.softmax_with_cross_entropy
