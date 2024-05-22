.. _cn_api_paddle_nn_AdaptiveLogSoftmaxWithLoss:

AdaptiveLogSoftmaxWithLoss
-------------------------------

.. py:class:: paddle.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, name=None)
自适应 softmax 是一种高效的策略，用于训练输出空间庞大的模型，尤其在标签分布显著不平衡的场合下效果显著。例如，在自然语言建模领域，单词出现的频率遵循 Zipf's law。Zipf's law: https://en.wikipedia.org/wiki/Zipf%27s_law

自适应 softmax 将标签按照频率划分为多个簇。每个簇包含的目标数量不同，且频率较低的标签所在的簇会采用较低维度的嵌入，这样做可以显著减少计算量。在每个训练的小批量中，只有当至少有一个目标标签出现时，相应的簇才会被计算。

这种方法的设计理念是，频繁访问的簇（如包含最常见标签的初始簇）应该具有较低的计算成本，这意味着这些簇应该只包含少量的标签。

对于参数`cutoffs`，应该是按升序排序的整数序列。它控制簇的数量和目标分配到簇的方式。例如，设置 cutoffs = [10, 100, 1000]意味着前 10 个目标将分配到自适应 softmax 的'head'，目标 11, 12, ..., 100 将分配到第一个簇，而目标 101, 102, ..., 1000 将分配到第二个簇，而目标 1001, 1002, ..., n_classes - 1 将分配到最后一个，第三个簇。

对于参数`div_value`，用于计算每个附加簇的大小，其值为:math:`\left\lfloor\frac{\texttt{in\_features}}{\texttt{div\_value}^{idx}}\right\rfloor`，其中 :math:`idx` 是簇索引（对于较不频繁的单词，簇索引较大，索引从 :math:`1` 开始）。

对于参数`head_bias`，如果设置为 True，将在自适应 softmax 的'head'上添加偏置项。详细信息请参阅论文：https://arxiv.org/abs/1609.04309 。



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
