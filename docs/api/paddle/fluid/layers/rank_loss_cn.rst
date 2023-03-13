.. _cn_api_fluid_layers_rank_loss:

rank_loss
-------------------------------

.. py:function::  paddle.fluid.layers.rank_loss(label, left, right, name=None)




该 OP 实现了 RankNet 模型中的排序损失层。RankNet 是一种文档对（pairwise）排序模型，训练样本由一对文档（假设用 A、B 来表示）组成。标签（假设用 P 来表示）表示 A 的排名是否高于 B。更多详情请参考：`RankNet <http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf>`_

排序损失层有三个输入：:math:`o_i` 、 :math:`o_j` 和 :math:`\tilde{P_{ij}}`，输入分别表示 RankNet 模型对文档 A、B 的输出得分和标签 P 的值；排序损失层的输入是批输入数据（批大小大于等于 1）；标签 P 的取值可以为：{0, 1} 或 {0, 0.5, 1}，其中，0.5 表示输入文档对排序相同。输入数据的排序损失 :math:`C_{i,j}` 计算过程如下：

.. math::

    C_{i,j} &= -\tilde{P_{ij}} * o_{i,j} + \log(1 + e^{o_{i,j}})

    o_{i,j} &=  o_i - o_j

    \tilde{P_{i,j}} &= \left \{0, 0.5, 1 \right \} \ or \ \left \{0, 1 \right \}

参数
::::::::::::

    - **label** (Variable)：维度为 :math:`[batch,1]` 的 2-D ``Tensor``，数据类型为 float32。其中 batch 表示批数据的大小。表示 A 的排名是否高于 B。
    - **left** (Variable)：维度为 :math:`[batch,1]` 的 2-D ``Tensor``，数据类型为 float32。其中 batch 表示批数据的大小。表示 RankNet 对文档 A 的输出得分。
    - **right** (Variable)：维度为 :math:`[batch,1]` 的 2-D ``Tensor``，数据类型为 float32。其中 batch 表示批数据的大小。表示 RankNet 对文档 B 的输出得分。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::
表示排序损失层输出值的 ``Tensor``，数据类型为 float32，返回值维度为 :math:`[batch,1]` 。

返回类型
::::::::::::
Variable

抛出异常
::::::::::::

    - ``ValueError`` - 输入 ``label`` ， ``left``，和 ``right`` 至少有一个不是 ``Variable`` 类型。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.rank_loss
