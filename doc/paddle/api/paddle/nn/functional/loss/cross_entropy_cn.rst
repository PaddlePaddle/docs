.. _cn_api_paddle_functional_cross_entropy:

cross_entropy
-------------------------------

.. py:function:: paddle.nn.functional.cross_entropy(input, label, weight=None, ignore_index=-100, reduction="mean", soft_label=False, axis=-1, name=None)

该OP实现了softmax交叉熵损失函数。该函数会将softmax操作、交叉熵损失函数的计算过程进行合并，从而提供了数值上更稳定的计算。

该OP默认会对结果进行求mean计算, 您也可以影响该默认行为， 具体参考reduction参数说明。

该OP可用于计算硬标签或软标签的交叉熵。其中，硬标签是指实际label值，例如：0, 1, 2...，软标签是指实际label的概率，例如：0.6, 0,8, 0,2... 

该OP的计算包括以下两个步骤：

- **一. softmax交叉熵**

1. 硬标签（每个样本仅可分到一个类别）

   .. math::
      \\loss_j=-\text{logits}_{label_j}+\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right) 
        j = 1,..., N为样本数, C为类别数

2. 软标签（每个样本以一定的概率被分配至多个类别中，概率和为1）

   .. math::
      \\loss_j=-\sum_{i=0}^{C}\text{label}_i\left(\text{logits}_i-\log\left(\sum_{i=0}^{C}\exp(\text{logits}_i)\right)\right)
        j = 1,...,N为样本数, C为类别数

- **二. weight及reduction处理**

1. weight情况

如果 ``weight`` 参数为 ``None`` ,则直接进入下一步。

如果 ``weight`` 参数不为 ``None`` , 则对每个样本的交叉熵进行weight加权(区分soft_label = False or True):

1.1. 硬标签情况(soft_label = False)

     .. math::
        \\loss_j=loss_j*weight[label_j] 

1.2. 软标签情况(soft_label = True)

     .. math::
        \\loss_j=loss_j*\sum_{i}\left(weight[label_i]*logits_i\right)

2. reduction情况

   (2.1) 如果 ``reduction`` 参数为 ``none``  

     则直接返回上一步结果

   (2.2) 如果 ``reduction`` 参数为 ``sum``  

     则返回上一步结果的和

     .. math::
        \\loss=\sum_{j}loss_j

   (2.3) 如果 ``reduction`` 参数为 ``mean``, 则根据``weight`` 参数情况进行处理:  

2.3.1. 如果 ``weight`` 参数为 ``None`` 

     则返回上一步结果的平均值

     .. math::
        \\loss=\sum_{j}loss_j/N, N为样本数

2.3.2. 如果 ``weight`` 参数不为 ``None`` , 则返回上一步结果的加权平均值

    (1) 硬标签情况(soft_label = False)

     .. math::
        \\loss=\sum_{j}loss_j/\sum_{j}weight[label_j] 

    (2)  软标签情况(soft_label = True)

     .. math::
        \\loss=\sum_{j}loss_j/\sum_{j}\left(\sum_{i}weight[label_i]\right)
 
参数
:::::::::
    - **input** (Tensor) – 维度为 :math:`[N_1, N_2, ..., N_k, C]` 的多维Tensor，其中最后一维C是类别数目。数据类型为float32或float64。它需要未缩放的 ``input`` 。该OP不应该对softmax运算的输出进行操作，否则会产生错误的结果。
    - **label** (Tensor) – 输入input对应的标签值。若soft_label=False，要求label维度为 :math:`[N_1, N_2, ..., N_k]` 或 :math:`[N_1, N_2, ..., N_k, 1]` ，数据类型为'int32', 'int64', 'float32', 'float64'，且值必须大于等于0且小于C；若soft_label=True，要求label的维度、数据类型与input相同，且每个样本各软标签的总和为1。
    - **weight** (Tensor, optional) – 权重张量，需要手动给每个类调整权重，形状是（C）。它的维度与类别相同，数据类型为float32，float64。默认值为None。
    - **ignore_index** (int) – 指定一个忽略的标签值，此标签值不参与计算，负值表示无需忽略任何标签值。仅在soft_label=False时有效。 默认值为-100。
    - **reduction** (str, optional) – 指示如何按批次大小平均损失，可选值为"none","mean","sum"，如果选择是"mean"，则返回reduce后的平均损失；如果选择是"sum"，则返回reduce后的总损失。如果选择是"none"，则返回没有reduce的损失。默认值是“mean”。
    - **soft_label** (bool, optional) – 指明label是否为软标签。默认为False，表示label为硬标签；若soft_label=True则表示软标签。
    - **axis** (int, optional) - 进行softmax计算的维度索引。 它应该在 :math:`[-1，dim-1]` 范围内，而 ``dim`` 是输入logits的维度。 默认值：-1。
    - **name** (str，optional） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

返回
:::::::::
表示交叉熵结果的Tensor，数据类型与input相同。若soft_label=False，则返回值维度与label维度相同；若soft_label=True，则返回值维度为 :math:`[N_1, N_2, ..., N_k, 1]` 。


代码示例
:::::::::

..  code-block:: python

        import paddle

        input_data = paddle.rand(shape=[5, 100])
        label_data = paddle.randint(0, 100, shape=[5,1], dtype="int64")
        weight_data = paddle.rand([100])

        loss = paddle.nn.functional.cross_entropy(input=input_data, label=label_data, weight=weight_data)
        print(loss)
        # [4.38418674]


