.. _cn_api_fluid_layers_cross_entropy:

cross_entropy
-------------------------------

.. py:function:: paddle.fluid.layers.cross_entropy(input, label, soft_label=False, ignore_index=-100)




该OP计算输入input和标签label间的交叉熵，可用于计算硬标签或软标签的交叉熵。

1. 硬标签（每个样本仅可分到一个类别）

     .. math::
        \\loss_j=-\text{logits}_{label_j}+\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right) , j = 1,...,N, N为样本数, C为类别数\\

2. 软标签（每个样本以一定的概率被分配至多个类别中，概率和为1）

     .. math::
        \\loss_j=-\sum_{i=0}^{C}\text{label}_i\left(\text{logits}_i-\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right)\right), j = 1,...,N, N为样本数, C为类别数\\

参数：
    - **input** (Tensor) – 维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维Tensor，其中最后一维D是类别数目。数据类型为float32或float64。
    - **label** (Tensor) – 输入input对应的标签值。若soft_label=False，要求label维度为 :math:`[N_1, N_2, ..., N_k]` 或 :math:`[N_1, N_2, ..., N_k, 1]` ，数据类型为int64，且值必须大于等于0且小于D；若soft_label=True，要求label的维度、数据类型与input相同，且每个样本各软标签的总和为1。
    - **soft_label** (bool) – 指明label是否为软标签。默认为False，表示label为硬标签；若soft_label=True则表示软标签。
    - **ignore_index** (int) – 指定一个忽略的标签值，此标签值不参与计算，负值表示无需忽略任何标签值。仅在soft_label=False时有效。 默认值为-100。

返回：Tensor, 表示交叉熵结果的Tensor，数据类型与input相同。若soft_label=False，则返回值维度与label维度相同；若soft_label=True，则返回值维度为 :math:`[N_1, N_2, ..., N_k, 1]` 。


**代码示例**

..  code-block:: python

        import paddle.fluid as fluid
        class_num = 7
        x = fluid.layers.data(name='x', shape=[3, 10], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        predict = fluid.layers.fc(input=x, size=class_num, act='softmax')
        cost = fluid.layers.cross_entropy(input=predict, label=label)
