.. _cn_api_paddle_distribution_Distribution:

Distribution
-------------------------------

.. py:class:: paddle.distribution.Distribution(batch_shape=(), event_shape=(), validate_args=None)

概率分布的抽象基类，在具体的分布中实现具体功能。

参数
:::::::::

- **batch_shape** - 概率分布参数批量形状。一元分布 ``batch_shape=param.shape``，多元分
  布 ``batch_shape=param.shape[:-1]``，其中 param 表示分布参数，支持 broadcast 语义。
- **event_shape** - 多元概率分布维数形状。一元分布 ``event_shape=()``，多元分布
  ``event_shape=param.shape[-1:]``，其中 param 表示分布参数，支持 broadcast 语义。
- **validate_args** (bool|None，可选) - 是否启用参数校验。默认值为 None。

属性
:::::::::

arg_constraints
'''''''''

返回该概率分布参数需要满足的约束条件。

**返回**

dict，分布参数与其约束条件的映射。

support
'''''''''

返回表示该概率分布支持集的约束对象。

**返回**

Constraint|None，表示支持集的约束对象。

mean
'''''''''

概率分布的均值。

**返回**

Tensor，均值。

mode
'''''''''

概率分布的众数。

**返回**

Tensor，众数。

variance
'''''''''

概率分布的方差。

**返回**

Tensor，方差。

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

sample_n(n)
''''''''''

从分布中生成 ``n`` 个样本。

**参数**

    - **n** (int) - 采样数量。

entropy()
'''''''''

分布的信息熵

log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

cdf(value)
''''''''''

计算 ``value`` 处的累计概率密度函数或累计概率质量函数值。

**参数**

    - **value** (Tensor) - 输入 Tensor。

icdf(value)
''''''''''

计算 ``value`` 处的逆累计概率密度函数或逆累计概率质量函数值。

**参数**

    - **value** (Tensor) - 输入 Tensor。

enumerate_support(expand=True)
''''''''''''''''''''''''''''''

返回离散概率分布支持集中的所有取值。

**参数**

    - **expand** (bool，可选) - 是否扩展结果 Tensor。默认值为 True。

perplexity()
''''''''''''

返回该概率分布的困惑度。

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

set_default_validate_args(value)
''''''''''''''''''''''''''''''''

设置是否默认启用参数校验。

**参数**

    - **value** (bool) - 是否默认启用参数校验。
