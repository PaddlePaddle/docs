.. _cn_api_paddle_nn_AdaptiveLogSoftmaxWithLoss:

AdaptiveLogSoftmaxWithLoss
-------------------------------

.. py:class:: paddle.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, name=None)


Efficient softmax approximation 的高效 softmax 逼近，正如 Edouard Grave、Armand Joulin、Moustapha Cissé、David Grangier 和 Hervé Jégou 在 Efficient softmax approximation for GPUs 一文中所述 https://arxiv.org/abs/1609.04309。

自适应 softmax 是一种用于训练具有大输出空间的模型的近似策略。当标签分布高度不平衡时，例如在自然语言建模中，其中单词频率分布大致遵循 Zipf's law_时，它最为有效。_Zipf's law: https://en.wikipedia.org/wiki/Zipf%27s_law

自适应 softmax 将标签分成几个簇，根据它们的频率。这些簇可能包含不同数量的目标。此外，包含较不频繁标签的簇为这些标签分配较低维度的嵌入，从而加速计算。对于每个小批量，仅评估至少存在一个目标的簇。

其思想是经常访问的簇（比如第一个簇，包含最频繁的标签），计算成本也应该较低，即包含少量分配的标签。我们建议查看原始论文以获取更多详细信息。

对于属性`cutoffs`，应该是按升序排序的整数序列。它控制簇的数量和目标分配到簇的方式。例如，设置 cutoffs = [10, 100, 1000]意味着前 10 个目标将分配到自适应 softmax 的'head'，目标 11, 12, ..., 100 将分配到第一个簇，而目标 101, 102, ..., 1000 将分配到第二个簇，而目标 1001, 1002, ..., n_classes - 1 将分配到最后一个，第三个簇。

对于属性`div_value`，用于计算每个附加簇的大小，其值为:math:`\left\lfloor\frac{\texttt{in\_features}}{\texttt{div\_value}^{idx}}\right\rfloor`，其中 :math:`idx` 是簇索引（对于较不频繁的单词，簇索引较大，索引从 :math:`1` 开始）。

对于属性`head_bias`，如果设置为 True，将在自适应 softmax 的'head'上添加偏置项。详细信息请参阅论文。在官方实现中设置为 False。


参数
:::::::::
    - **in_features** (int): 输入 tensor 的特征数量。
    - **n_classes** (int): 数据集中类型的个数。
    - **cutoffs** (Sequence): 用于将 label 分配到不同存储桶的截断值。
    - **div_value** (float, 可选): 用于计算簇大小的指数值. 默认值：4.0。
    - **head_bias** (bool, 可选): 如果为 ``True``，向自适应 softmax 的头部添加偏置项. 默认值：``False``.
    - **name** (str, 可选): 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **input** (Tensor): - 输入 Tensor，形状为[N, in_features]，N 是批尺寸。
    - **label** (Tensor): - 目标值，形状为[N]。
    - **output1** (Tensor): - 形状为[N]。
    - **output2** (Scalar): - 标量，无形状

返回
:::::::::
用于计算自适应 softmax 的可调用对象。

代码示例
:::::::::
COPY-FROM: paddle.nn.AdaptiveLogSoftmaxWithLoss
