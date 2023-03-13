.. _cn_api_fluid_layers_cross_entropy:

cross_entropy
-------------------------------

.. py:function:: paddle.fluid.layers.cross_entropy(input, label, soft_label=False, ignore_index=-100)




该 OP 计算输入 input 和标签 label 间的交叉熵，可用于计算硬标签或软标签的交叉熵。

1. 硬标签交叉熵算法：若 soft_label = False, :math:`label[i_1, i_2, ..., i_k]` 表示每个样本的硬标签值：

     .. math::
        \\output[i_1, i_2, ..., i_k]=-log(input[i_1, i_2, ..., i_k, j]), label[i_1, i_2, ..., i_k] = j, j != ignore\_index\\

2. 软标签交叉熵算法：若 soft_label = True, :math:`label[i_1, i_2, ..., i_k, j]` 表明每个样本对应类别 j 的软标签值：

     .. math::
        \\output[i_1, i_2, ..., i_k]= -\sum_{j}label[i_1,i_2,...,i_k,j]*log(input[i_1, i_2, ..., i_k,j])\\

参数
::::::::::::

    - **input** (Tensor) – 维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维 Tensor，其中最后一维 D 是类别数目。数据类型为 float32 或 float64。
    - **label** (Tensor) – 输入 input 对应的标签值。若 soft_label=False，要求 label 维度为 :math:`[N_1, N_2, ..., N_k]` 或 :math:`[N_1, N_2, ..., N_k, 1]`，数据类型为 int64，且值必须大于等于 0 且小于 D；若 soft_label=True，要求 label 的维度、数据类型与 input 相同，且每个样本各软标签的总和为 1。
    - **soft_label** (bool) – 指明 label 是否为软标签。默认为 False，表示 label 为硬标签；若 soft_label=True 则表示软标签。
    - **ignore_index** (int) – 指定一个忽略的标签值，此标签值不参与计算，负值表示无需忽略任何标签值。仅在 soft_label=False 时有效。默认值为-100。

返回
::::::::::::
Tensor，表示交叉熵结果的 Tensor，数据类型与 input 相同。若 soft_label=False，则返回值维度与 label 维度相同；若 soft_label=True，则返回值维度为 :math:`[N_1, N_2, ..., N_k, 1]` 。


代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.cross_entropy
