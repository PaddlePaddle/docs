.. _cn_api_paddle_io_SubsetRandomSampler:

SubsetRandomSampler
-------------------------------

.. py:class:: paddle.io.SubsetRandomSampler(indices)

从给定的索引列表中随机采样元素，而不进行替换

参数
:::::::::

    - **indices** (tuple|list) - 子集在原数据集中的索引序列，需要是 list 或者 tuple 类型。

返回
:::::::::
SubsetRandomSampler，返回根据权重随机采样下标的采样器



代码示例
:::::::::

COPY-FROM: paddle.io.SubsetRandomSampler
