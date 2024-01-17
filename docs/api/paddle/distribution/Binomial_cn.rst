.. _cn_api_paddle_distribution_Binomial:

Binomial
-------------------------------

.. py:class:: paddle.distribution.Binomial(total_count, probs)


在概率论和统计学中，Binomial 是一种最基本的离散型概率分布，定义在 :math:`[0, n] \cap \mathbb{N}` 上，可以看作是扔一枚硬币（可能是不公平的硬币）被扔出正面的次数，
它的随机变量的取值可以看作是一系列独立的伯努利实验结果的总和。

其概率质量函数（pmf）为：

.. math::

    pmf(x; n, p) = \frac{n!}{x!(n-x)!}p^{x}(1-p)^{n-x}

其中：

- :math:`n` 表示伯努利实验次数。
- :math:`p` 表示每次伯努利实验中事件发生的概率。

参数
:::::::::

    - **total_count** (int|Tensor) - 即上述公式中 :math:`n` 参数，大于零，表示伯努利实验次数。如果 :attr:`total_count` 的输入数据类型是 int 则会被转换
      成数据类型为 paddle 全局默认数据类型的 1-D Tensor，否则将转换成与 :attr:`probs` 相同的数据类型。

    - **probs** (float|Tensor) - 即上述公式中 :math:`p` 参数，在 [0, 1] 区间内，表示每次伯努利实验中事件发生的概率。如果 :attr:`probs` 的输
      入数据类型是 float 则会被转换为 paddle 全局默认数据类型的 1-D Tensor。


代码示例
:::::::::

COPY-FROM: paddle.distribution.Binomial

属性
:::::::::

mean
'''''''''

Binomial 分布的均值

**返回**

Tensor，均值

variance
'''''''''

Binomial 分布的方差

**返回**

Tensor，方差

方法
:::::::::

prob(value)
'''''''''

计算 value 的概率。

**参数**

    - **value** (Tensor) - 待计算值。

**返回**

Tensor，value 的概率。数据类型与 :attr:`probs` 相同。


log_prob(value)
'''''''''

计算 value 的对数概率。

**参数**

    - **value** (Tensor) - 待计算值。

**返回**

Tensor，value 的对数概率。数据类型与 :attr:`probs` 相同。


sample()
'''''''''

从 Binomial 分布中生成满足特定形状的样本数据。最终生成样本形状为 ``shape+batch_shape`` 。

**参数**

    - **shape** (Sequence[int]，可选)：采样次数。

**返回**

Tensor：样本数据。其维度为 :math:`\text{sample shape} + \text{batch shape}` 。

entropy()
'''''''''

计算 Binomial 分布的信息熵。

.. math::

    \mathcal{H}(X) = - \sum_{x \in \Omega} p(x) \log{p(x)}

**返回**

类别分布的信息熵，数据类型与 :attr:`probs` 相同。

kl_divergence(other)
'''''''''

相对于另一个类别分布的 KL 散度，两个分布需要有相同的 :attr:`total_count`。

**参数**

    - **other** (Binomial) - 输入的另一个类别分布。

**返回**

相对于另一个类别分布的 KL 散度，数据类型与 :attr:`probs` 相同。
