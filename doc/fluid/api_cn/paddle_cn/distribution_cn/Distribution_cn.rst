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
    - **value** (Variable) - 输入张量。数据类型为float32或float64。

.. py:function:: probs(value)

概率密度函数

参数：
    - **value** (Variable) - 输入张量。数据类型为float32或float64。

.. py:function:: kl_divergence(other)

两个分布之间的KL散度。

参数：
    - **other** (Distribution) - Distribution的实例。

.. py:function:: _validate_args(*args)

对分布的参数进行验证。

参数：
    - **value** (float, list, numpy.ndarray, Variable) - 输入的数据类型。
    
Raises ValueError：若其中一个参数为Variable，那么所有的参数都应该为Variable

.. py:function:: _to_variable(*args)

将参数args转化为Variable数据类型。

参数：
    - **value** (float, list, numpy.ndarray, Variable) - 输入的数据类型。
    
返回：参数的Variable类型






