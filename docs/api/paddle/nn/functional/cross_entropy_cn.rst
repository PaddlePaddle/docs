.. _cn_api_paddle_nn_functional_cross_entropy:

cross_entropy
-------------------------------

.. py:function:: paddle.nn.functional.cross_entropy(input, label, weight=None, ignore_index=-100, reduction="mean", soft_label=False, axis=-1, name=None)

实现了 softmax 交叉熵损失函数。该函数会将 softmax 操作、交叉熵损失函数的计算过程进行合并，从而提供了数值上更稳定的计算。

默认会对结果进行求 mean 计算，您也可以影响该默认行为，具体参考 reduction 参数说明。

可用于计算硬标签或软标签的交叉熵。其中，硬标签是指实际 label 值，例如：0, 1, 2...，软标签是指实际 label 的概率，例如：0.6, 0,8, 0,2..。

计算包括以下两个步骤：

- **一、softmax 交叉熵**

1. 硬标签（每个样本仅可分到一个类别）

   .. math::
      \\loss_j=-\text{logits}_{label_j}+\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right)
        , j = 1,...,N, N 为样本数，C 为类别数

2. 软标签（每个样本以一定的概率被分配至多个类别中，概率和为 1）

   .. math::
      \\loss_j=-\sum_{i=0}^{C}\text{label}_i\left(\text{logits}_i-\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right)\right)
        , j = 1,...,N, N 为样本数，C 为类别数

- **二、weight 及 reduction 处理**

1. weight 情况

如果 ``weight`` 参数为 ``None``，则直接进入下一步。

如果 ``weight`` 参数不为 ``None``，则对每个样本的交叉熵进行 weight 加权(区分 soft_label = False or True):

1.1 硬标签情况(soft_label = False)

     .. math::
        \\loss_j=loss_j*weight[label_j]

1.2 软标签情况(soft_label = True)

     .. math::
        \\loss_j=loss_j*\sum_{i}\left(weight[label_i]*logits_i\right)

2. reduction 情况

2.1 如果 ``reduction`` 参数为 ``none``

     则直接返回上一步结果

2.2 如果 ``reduction`` 参数为 ``sum``

     则返回上一步结果的和

     .. math::
        \\loss=\sum_{j}loss_j

2.3 如果 ``reduction`` 参数为 ``mean``，则根据 ``weight``  参数情况进行处理：

2.3.1 如果 ``weight`` 参数为 ``None``

     则返回上一步结果的平均值

     .. math::
        \\loss=\sum_{j}loss_j/N, N 为样本数

2.3.2 如果 ``weight`` 参数不为 ``None``，则返回上一步结果的加权平均值

    (1) 硬标签情况(soft_label = False)

     .. math::
        \\loss=\sum_{j}loss_j/\sum_{j}weight[label_j]

    (2)  软标签情况(soft_label = True)

     .. math::
        \\loss=\sum_{j}loss_j/\sum_{j}\left(\sum_{i}weight[label_i]\right)

参数
:::::::::
    - **input** (Tensor) - 维度为 :math:`[N_1, N_2, ..., N_k, C]` 的多维 Tensor，其中最后一维 C 是类别数目。数据类型为 float32 或 float64。它需要未缩放的 ``input``。该 OP 不应该对 softmax 运算的输出进行操作，否则会产生错误的结果。
    - **label** (Tensor) - 输入 input 对应的标签值。若 soft_label=False，要求 label 维度为 :math:`[N_1, N_2, ..., N_k]` 或 :math:`[N_1, N_2, ..., N_k, 1]`，数据类型为'int32', 'int64', 'float32', 'float64'，且值必须大于等于 0 且小于 C；若 soft_label=True 且没有指定 label_smoothing ，要求 label 的维度、数据类型与 input 相同，且每个样本各软标签的总和为 1；若指定了 label_smoothing (label_smoothing > 0.0) 时，无论 soft_label 是什么值，label 的维度和数据类型可以是前面两种情况中的任意一种。换句话说，如果 label_smoothing > 0.0，label 可以是独热标签或整数标签。
    - **weight** (Tensor，可选) - 权重 Tensor，需要手动给每个类调整权重，形状是（C）。它的维度与类别相同，数据类型为 float32，float64。默认值为 None。
    - **ignore_index** (int) - 指定一个忽略的标签值，此标签值不参与计算，负值表示无需忽略任何标签值。仅在 soft_label=False 时有效。默认值为-100。
    - **reduction** (str，可选) - 指示如何按批次大小平均损失，可选值为"none","mean","sum"，如果选择是"mean"，则返回 reduce 后的平均损失；如果选择是"sum"，则返回 reduce 后的总损失。如果选择是"none"，则返回没有 reduce 的损失。默认值是“mean”。
    - **soft_label** (bool，可选) - 指明 label 是否为软标签。默认为 False，表示 label 为硬标签；若 soft_label=True 则表示软标签。
    - **label_smoothing** （float，可选）- 指定计算损失时的标签平滑度，它应该在 :math:`[0.0，1.0]` 范围内。其中 0.0 表示无平滑。使得平滑后的标签变成原始真实标签和均匀分布的混合，默认值： 0.0。
    - **axis** (int，可选) - 进行 softmax 计算的维度索引。它应该在 :math:`[-1，dim-1]` 范围内，而 ``dim`` 是输入 logits 的维度。默认值：-1。
    - **use_softmax** (bool，可选) - 指定是否对 input 进行 softmax 归一化。默认值：True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
表示交叉熵结果的 Tensor，数据类型与 input 相同。若 soft_label=False，则返回值维度与 label 维度相同；若 soft_label=True，则返回值维度为 :math:`[N_1, N_2, ..., N_k, 1]` 。


代码示例
:::::::::

COPY-FROM: paddle.nn.functional.cross_entropy
