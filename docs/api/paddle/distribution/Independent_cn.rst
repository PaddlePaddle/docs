.. _cn_api_paddle_distribution_Independent:

Independent
-------------------------------

.. py:class:: paddle.distribution.Independent(base, reinterpreted_batch_rank)

将一个基础分布 ``base`` 的最右侧 ``reinterpreted_batch_rank`` 批维度转换为事件维度。


参数
:::::::::

- **base** (Distribution) - 基础分布。
- **reinterpreted_batch_rank** (int） - 用于转换为事件维度的批维度数量。

代码示例
:::::::::

COPY-FROM: paddle.distribution.Independent


方法
:::::::::

property mean
'''''''''

计算分布均值。


property variance
'''''''''

计算分布方差。


prob(value)
'''''''''

计算 value 的概率。

**参数**

- **value** (Tensor) - 待计算值。

**返回**

- Tensor: value 的概率。


log_prob(value)
'''''''''

计算 value 的对数概率。

**参数**

- **value** (Tensor) - 待计算值。

**返回**

- Tensor: value 的对数概率。


sample(shape=[])
'''''''''

从 Beta 分布中生成满足特定形状的样本数据。

**参数**

- **shape** (Sequence[int]，可选)：采样次数。最终生成样本形状为 ``shape+batch_shape`` 。

**返回**

- Tensor：样本数据。

entropy()
'''''''''

计算信息熵。
