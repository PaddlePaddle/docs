.. _cn_api_paddle_distribution_Poisson:

Poisson
-------------------------------

.. py:class:: paddle.distribution.Poisson(rate)


在概率论和统计学中，Poisson 是一种最基本的离散型概率分布，定义在非负整数集上，用来描述单位时间内随机事件发生次数的概率分布。

其概率质量函数（pmf）为：

.. math::

    pmf(x; \lambda) = \frac{e^{-\lambda} \cdot \lambda^x}{x!}

其中，:math:`\lambda` 表示事件平均发生率。


参数
:::::::::

    - **rate** (int|float|Tensor) - 即上述公式中 :math:`\lambda` 参数，大于零，表示事件平均发生率，即单位时间内的事件发生次数。如果输入数据类型不是 int 或 float
      :attr:`rate` 的数据类型会被转换成数据类型为 paddle 全局默认数据类型的 1-D Tensor

代码示例
:::::::::

COPY-FROM: paddle.distribution.Poisson

属性
:::::::::

mean
'''''''''

Poisson 分布的均值

**返回**

Tensor，均值

variance
'''''''''

Poisson 分布的方差

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

Tensor，value 的概率。数据类型与 :attr:`rate` 相同。


log_prob(value)
'''''''''

计算 value 的对数概率。

**参数**

    - **value** (Tensor) - 待计算值。

**返回**

Tensor，value 的对数概率。数据类型与 :attr:`rate` 相同。


sample()
'''''''''

从 Poisson 分布中生成满足特定形状的样本数据。最终生成样本形状为 ``shape+batch_shape`` 。

**参数**

    - **shape** (Sequence[int]，可选)：采样次数。

**返回**

Tensor，样本数据。其维度为 :math:`\text{sample shape} + \text{batch shape}` 。

entropy()
'''''''''

计算 Poisson 分布的信息熵。

.. math::

    \mathcal{H}(X) = - \sum_{x \in \Omega} p(x) \log{p(x)}

**返回**

类别分布的信息熵，数据类型与 :attr:`rate` 相同。

kl_divergence(other)
'''''''''

相对于另一个类别分布的 KL 散度，两个分布需要有相同的 :math:`\text{batch shape}`。

**参数**

    - **other** (Poisson) - 输入的另一个类别分布。

**返回**

相对于另一个类别分布的 KL 散度，数据类型与 :attr:`rate` 相同。
