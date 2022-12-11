.. _cn_api_paddle_distribution_TransformedDistribution:

TransformedDistribution
-------------------------------

基于一个基础分布和一系列分布变换构建一个新的分布。

.. py:class:: paddle.distribution.TransformedDistribution(base, transforms)

参数
:::::::::

- **base** (Distribution) - 基础分布。
- **transforms** (Sequence[Transform]） - 变换序列。

代码示例
:::::::::

COPY-FROM: paddle.distribution.TransformedDistribution

方法
:::::::::


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


sample(shape=())
'''''''''

生成满足特定形状的样本数据。

**参数**

- **shape** (Sequence[int]，可选)：采样形状。

**返回**

- Tensor：样本数据。

rsample(shape=())
'''''''''

重参数化采样，生成满足特定形状的样本数据。

**参数**

- **shape** (Sequence[int]，可选)：采样形状。

**返回**

- Tensor：样本数据。
