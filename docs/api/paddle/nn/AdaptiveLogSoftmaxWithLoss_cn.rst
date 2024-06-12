.. _cn_api_paddle_nn_AdaptiveLogSoftmaxWithLoss:

AdaptiveLogSoftmaxWithLoss
-------------------------------

.. py:class:: paddle.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4.0, head_bias=False, name=None)
AdaptiveLogSoftmaxWithLoss是一种高效的策略，通常用于自然语言处理任务中的语言模型训练，尤其是在处理具有大量词汇且标签分布显著不平衡的语料库时。

AdaptiveLogSoftmaxWithLoss将标签按照频率划分为多个组，每个组包含的目标数量不同，且在频率较低的标签所在的组中会采用较低维度的嵌入，从而显著减少计算量。

在每个训练的小批量中，只有当至少有一个目标标签出现时，相应的组才会被计算。这种方法的设计理念是，频繁访问的组（如包含最常见标签的初始组）应该具有较低的计算成本。

对于参数 ``cutoffs``，按升序排序的整数序列。它控制组的数量和目标分配到组的方式。例如，设置 ``cutoffs = [10, 100, 1000]`` 意味着前 10 个目标将分配到AdaptiveLogSoftmaxWithLoss的 ``head``，目标 11, 12, ..., 100 将分配到第一个组，而目标 101, 102, ..., 1000 将分配到第二个组，而目标 1001, 1002, ..., n_classes - 1 将分配到第三个组。

对于参数 ``div_value``，用于计算每个附加组的大小，其值为 :math:`\left\lfloor \frac{\text{in\_features}}{\text{div\_value}^{\text{idx}}} \right\rfloor`，其中 ``idx`` 是组索引（对于较不频繁的单词，组索引较大，索引从 :math:`1` 开始）。

对于参数 ``head_bias``，如果设置为 True，将在AdaptiveLogSoftmaxWithLoss的 ``head`` 上添加偏置项。详细信息请参阅论文：https://arxiv.org/abs/1609.04309 。



参数
:::::::::
    - **in_features** (int): 输入 Tensor 的特征数量。
    - **n_classes** (int): 数据集中类型的个数。
    - **cutoffs** (Sequence): 用于将 label 分配到不同存储组的截断值。
    - **div_value** (float, 可选): 用于计算组大小的指数值。默认值：4.0。
    - **head_bias** (bool, 可选): 如果为 ``True``，AdaptiveLogSoftmaxWithLoss的 ``head`` 添加偏置项。默认值： ``False``.
    - **name** (str, 可选): 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **input** (Tensor): - 输入 Tensor，形状为 ``[N, in_features]``， ``N`` 是批尺寸。
    - **label** (Tensor): - 目标值，形状为 ``[N]``。
    - **output1** (Tensor): - 形状为 ``[N]``。
    - **output2** (Scalar): - 标量，无形状

返回
:::::::::
用于计算自适应 softmax 的可调用对象。

代码示例
:::::::::
COPY-FROM: paddle.nn.AdaptiveLogSoftmaxWithLoss
