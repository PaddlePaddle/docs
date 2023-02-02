.. _cn_api_nn_loss_CrossEntropyLoss:

CrossEntropyLoss
-------------------------------

.. py:function:: paddle.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', soft_label=False, axis=-1, name=None)

计算输入 input 和标签 label 间的交叉熵损失，它结合了 `LogSoftmax` 和 `NLLLoss` 的函数计算，可用于训练一个 `n` 类分类器。

如果提供 `weight` 参数的话，它是一个 `1-D` 的 tensor，每个值对应每个类别的权重。
该损失函数的数学计算公式如下：

    .. math::
        loss_j =  -\text{input[class]} +
        \log\left(\sum_{i=0}^{K}\exp(\text{input}_i)\right), j = 1,..., K

当 `weight` 不为 `none` 时，损失函数的数学计算公式为：

    .. math::
        loss_j =  \text{weight[class]}(-\text{input[class]} +
        \log\left(\sum_{i=0}^{K}\exp(\text{input}_i)\right)), j = 1,..., K


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

COPY-FROM: paddle.nn.CrossEntropyLoss
