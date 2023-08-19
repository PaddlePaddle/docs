.. _cn_api_nn_loss_CrossEntropyLoss:

CrossEntropyLoss
-------------------------------

.. py:function:: paddle.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', soft_label=False, axis=-1, name=None)

默认情况下， CrossEntropyLoss 使用 softmax 实现（即 :use_softmax=True ）。该函数结合了 softmax 操作的计算和交叉熵损失函数，以提供更稳定的数值计算。

当 use_softmax=False 时，仅计算交叉熵损失函数而不使用 softmax。默认情况下，计算结果的平均值。可以使用 reduction 参数来影响默认行为。请参考参数部分了解详情。

可以用于计算带有 soft labels 和 hard labels 的 softmax 交叉熵损失。其中，hard labels 表示实际的标签值，例如 0、1、2 等。而 soft labels 表示实际标签的概率，例如 0.6、0.8、0.2 等。

计算包括以下两个步骤:

-  **I.softmax 交叉熵**

    1. Hard label (每个样本仅表示为一个类别)

    1.1. 当 use_softmax=True 时

        .. math::
          \\loss_j=-\text{logits}_{label_j}+\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right) , j = 1,...,N

        其中，N 是样本数，C 是类别数。

    1.2. 当 use_softmax=False 时
        .. math::
          \\loss_j=-\log\left({P}_{label_j}\right) , j = 1,...,N

        其中，N 是样本数，C 是类别数，P 是输入（softmax 的输出）。


    2. Soft label (每个样本为多个类别分配一定的概率，概率和为 1).

    2.1. 当 use_softmax=True 时

        .. math::
          \\loss_j=-\sum_{i=0}^{C}\text{label}_i\left(\text{logits}_i-\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right)\right) , j = 1,...,N

        其中，N 是样本数，C 是类别数。

    2.2. 当 use_softmax=False 时

        .. math::
          \\loss_j=-\sum_{j=0}^{C}\left({label}_j*\log\left({P}_{label_j}\right)\right) , j = 1,...,N

        其中，N 是样本数，C 是类别数，P 是输入（softmax 的输出）。



-  **II.Weight 和 reduction 处理**

    1. Weight

        如果 ``weight`` 参数为 ``None`` , 直接进行下一步.

        如果 ``weight`` 参数不为 ``None`` , 每个样本的交叉熵按权重加权
        根据 soft_label = False 或 True 如下：

        1.1. Hard labels (soft_label = False)

        .. math::
            \\loss_j=loss_j*weight[label_j]


        1.2. Soft labels (soft_label = True)

         .. math::
            \\loss_j=loss_j*\sum_{i}\left(weight[label_i]*logits_i\right)

    2. reduction

        2.1 如果 ``reduction`` 参数为 ``none``

        直接返回之前的结果

        2.2 如果 ``reduction`` 参数为 ``sum``

        返回之前结果的和

        .. math::
           \\loss=\sum_{j}loss_j

        2.3 如果 ``reduction`` 参数为 ``mean`` , 则按照 ``weight`` 参数进行如下处理。

        2.3.1. 如果  ``weight``  参数为 ``None``

        返回之前结果的平均值

         .. math::
            \\loss=\sum_{j}loss_j/N

        其中，N 是样本数，C 是类别数。

        2.3.2. 如果“weight”参数不为“None”，则返回之前结果的加权平均值

        1. Hard labels (soft_label = False)

         .. math::
            \\loss=\sum_{j}loss_j/\sum_{j}weight[label_j]

        2. Soft labels (soft_label = True)

         .. math::
            \\loss=\sum_{j}loss_j/\sum_{j}\left(\sum_{i}weight[label_i]\right)


参数
:::::::::
    - **weight** (Tensor，可选) - 指定每个类别的权重。其默认为 `None`。如果提供该参数的话，维度必须为 `C` （类别数）。数据类型为 float32 或 float64。
    - **ignore_index** (int64，可选) - 指定一个忽略的标签值，此标签值不参与计算，负值表示无需忽略任何标签值。仅在 soft_label=False 时有效。默认值为-100。数据类型为 int64。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，数据类型为 string，可选值有：`none`, `mean`, `sum`。默认为 `mean`，计算 `mini-batch` loss 均值。设置为 `sum` 时，计算 `mini-batch` loss 的总和。设置为 `none` 时，则返回 loss Tensor。
    - **soft_label** (bool，可选) – 指明 label 是否为软标签。默认为 False，表示 label 为硬标签；若 soft_label=True 则表示软标签。
    - **axis** (int，可选) - 进行 softmax 计算的维度索引。它应该在 :math:`[-1，dim-1]` 范围内，而 ``dim`` 是输入 logits 的维度。默认值：-1。
    - **use_softmax** (bool，可选) - 指定是否对 input 进行 softmax 归一化。默认值：True。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
形状
:::::::::
    - **input** (Tensor): 输入 `Tensor`，数据类型为 float32 或 float64。其形状为 :math:`[N, C]`，其中 `C` 为类别数。对于多维度的情形下，它的形状为 :math:`[N, d_1, d_2, ..., d_k, C]` ，k >= 1。
    - **label** (Tensor): 当 soft_label=False 时，输入 input 对应的标签值，数据类型为 int64。其形状为 :math:`[N]`，每个元素符合条件：0 <= label[i] <= C-1。对于多维度的情形下，它的形状为 :math:`[N, d_1, d_2, ..., d_k]` ，k >= 1；当 soft_label=True 时，输入形状应与 input 一致，数据类型为 float32 或 float64 且每个样本的各标签概率和应为 1。
    - **output** (Tensor): 计算 `CrossEntropyLoss` 交叉熵后的损失值。


代码示例
:::::::::

COPY-FROM: paddle.nn.CrossEntropyLoss:code-example1
COPY-FROM: paddle.nn.CrossEntropyLoss:code-example2
