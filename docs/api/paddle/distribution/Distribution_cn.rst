.. _cn_api_distribution_Distribution:

Distribution
-------------------------------

.. py:class:: paddle.distribution.Distribution()




概率分布的抽象基类，在具体的分布中实现具体功能。


.. py:function:: sample()

从分布中采样

.. py:function:: entropy()

分布的信息熵

.. py:function:: log_prob(value)

对数概率密度函数

参数：
    - **value** (Tensor) - 输入张量。

.. py:function:: probs(value)

概率密度函数

参数：
    - **value** (Tensor) - 输入张量。

.. py:function:: kl_divergence(other)

两个分布之间的KL散度。

参数：
    - **other** (Distribution) - Distribution的实例。








