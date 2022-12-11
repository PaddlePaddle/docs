.. _cn_api_fluid_layers_cross_entropy:

cross_entropy
-------------------------------

.. py:function:: paddle.fluid.layers.cross_entropy(input, label, soft_label=False, ignore_index=-100)




该OP计算输入input和标签label间的交叉熵，可用于计算硬标签或软标签的交叉熵。

1. 硬标签交叉熵算法：若soft_label = False, :math:`label[i_1, i_2, ..., i_k]` 表示每个样本的硬标签值：

     .. math::
        \\output[i_1, i_2, ..., i_k]=-log(input[i_1, i_2, ..., i_k, j]), label[i_1, i_2, ..., i_k] = j, j != ignore\_index\\

2. 软标签交叉熵算法：若soft_label = True, :math:`label[i_1, i_2, ..., i_k, j]` 表明每个样本对应类别j的软标签值：

     .. math::
        \\output[i_1, i_2, ..., i_k]= -\sum_{j}label[i_1,i_2,...,i_k,j]*log(input[i_1, i_2, ..., i_k,j])\\

参数
::::::::::::

    - **input** (Tensor) – 维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维Tensor，其中最后一维D是类别数目。数据类型为float32或float64。
    - **label** (Tensor) – 输入input对应的标签值。若soft_label=False，要求label维度为 :math:`[N_1, N_2, ..., N_k]` 或 :math:`[N_1, N_2, ..., N_k, 1]`，数据类型为int64，且值必须大于等于0且小于D；若soft_label=True，要求label的维度、数据类型与input相同，且每个样本各软标签的总和为1。
    - **soft_label** (bool) – 指明label是否为软标签。默认为False，表示label为硬标签；若soft_label=True则表示软标签。
    - **ignore_index** (int) – 指定一个忽略的标签值，此标签值不参与计算，负值表示无需忽略任何标签值。仅在soft_label=False时有效。默认值为-100。

返回
::::::::::::
Tensor，表示交叉熵结果的Tensor，数据类型与input相同。若soft_label=False，则返回值维度与label维度相同；若soft_label=True，则返回值维度为 :math:`[N_1, N_2, ..., N_k, 1]` 。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.cross_entropy