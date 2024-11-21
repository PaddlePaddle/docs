.. _cn_api_paddle_distribution_Distribution:

Distribution
-------------------------------

.. py:class:: paddle.distribution.Distribution()

概率分布的抽象基类，在具体的分布中实现具体功能。

参数
:::::::::

- **batch_shape** - 概率分布参数批量形状。一元分布 ``batch_shape=param.shape``，多元分
  布 ``batch_shape=param.shape[:-1]``，其中 param 表示分布参数，支持 broadcast 语义。
- **event_shape** - 多元概率分布维数形状。一元分布 ``event_shape=()``，多元分布
  ``event_shape=param.shape[-1:]``，其中 param 表示分布参数，支持 broadcast 语义。


方法
:::::::::

sample(shape=[])
'''''''''

从分布中采样

**参数**

    - **shape** (Sequence[int]，可选) - 采样的样本维度。

rsample(shape=[])
'''''''''

从分布中重参数化采样

**参数**

    - **shape** (Sequence[int]，可选) - 重参数化采样的样本维度。

entropy()
'''''''''

分布的信息熵

log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

probs(value)
'''''''''

概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

kl_divergence(other)
'''''''''

两个分布之间的 KL 散度。

**参数**

    - **other** (Distribution) - Distribution 的实例。
