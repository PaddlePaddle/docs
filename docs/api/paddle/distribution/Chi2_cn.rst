.. _cn_api_paddle_distribution_Chi2:

Chi2
-------------------------------
.. py:class:: paddle.distribution.Chi2(df)


卡方分布是基于标准正态分布随机变量的平方和定义的一种连续概率分布。如果有 *k* 个独立的标准正态分布随机变量
:math:`Z_1, Z_2, \ldots, Z_k`, 那么这些随机变量的平方和
:math:`X = Z_1^2 + Z_2^2 + \cdots + Z_k^2`

服从自由度为 *k* 的卡方分布，记作 
:math:`X \sim \chi^2(k)`



参数
::::::::::::

    - **df** (float|Tensor) - 参数表示自由度，该值必须大于零。支持 Broadcast 语义。当参数类型为 Tensor 时，表示批量创建多个不同参数的分布，``batch_shape`` (参考 :ref:`cn_api_paddle_distribution_Distribution` 基类) 为参数。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.Chi2


属性
:::::::::

mean
'''''''''
卡方分布的均值。


variance
'''''''''
卡方分布的方差。


方法
:::::::::

prob(value)
'''''''''
卡方分布的概率密度函数。

**参数**

    - **value** (float|Tensor) - 输入值。


**返回**

    - **Tensor** - value 对应的概率密度。


log_prob(value)
'''''''''
卡方分布的对数概率密度函数。

**参数**

    - **value** (float|Tensor) - 输入值。

**返回**

    - **Tensor** - value 对应的对数概率密度。


entropy()
'''''''''
卡方分布的信息熵。

**返回**

    - Tensor: 信息熵。


kl_divergence(other)
'''''''''
两个卡方分布之间的 KL 散度。

**参数**

    - **other** (Geometric) - Chi2 的实例。

**返回**

    - Tensor: 两个卡方分布之间的 KL 散度。


sample(shape)
'''''''''
随机采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int], optional) - 采样的样本维度。

**返回**

    - **Tensor** - 指定维度的样本数据。数据类型为 float32。


rsample(shape)
'''''''''
重参数化采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int], optional) - 重参数化采样的样本维度。

**返回**

    - **Tensor** - 指定维度的样本数据。数据类型为 float32。
