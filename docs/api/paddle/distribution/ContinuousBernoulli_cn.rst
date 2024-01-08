.. _cn_api_paddle_distribution_ContinuousBernoulli:

ContinuousBernoulli
-------------------------------

.. py:class:: paddle.distribution.ContinuousBernoulli(probs, lims=(0.499, 0.501))


ContinuousBernoulli 是一种定义在 [0，1] 区间上的连续型概率分布，参数 :attr:`probs` 描述了其概率密度函数的形状。它可以被视为连续型的伯努利分布。
出自 `The continuous Bernoulli: fixing a pervasive error in variational autoencoders. <https://arxiv.org/abs/1907.06845>`_

其概率密度函数（pdf）为：

.. math::

    p(x;\lambda) = C(\lambda)\lambda^x (1-\lambda)^{1-x}

其中：

- :math:`x` 在 [0, 1] 区间内是连续的。
- :math:`\lambda` 表示事件发生的概率。
- :math:`C(\lambda)` 表示归一化常数因子，表达式如下：

.. math::

    {   C(\lambda) =
        \left\{
        \begin{aligned}
        &2 & \text{ if $\lambda = \frac{1}{2}$} \\
        &\frac{2\tanh^{-1}(1-2\lambda)}{1 - 2\lambda} & \text{ otherwise}
        \end{aligned}
        \right. }

参数
:::::::::

    - **probs** (int|float|Tensor) - 即上述公式中 :math:`\lambda` 参数，在 [0, 1] 内，刻画 ContinuousBernoulli 分布的
      概率密度函数的形状。如果 :attr:`probs` 的输入数据类型是 `int` 或 `float` ，则会被转换为数据类型为 paddle 全局默认数据类型的 1-D Tensor。

    - **lims** (tuple，可选) - 表示概率计算非稳定区域的区域宽度，非稳定区域的概率计算使用泰勒展开做近似。默认值为 (0.499, 0.501)。

代码示例
:::::::::

COPY-FROM: paddle.distribution.ContinuousBernoulli

属性
:::::::::

mean
'''''''''

ContinuousBernoulli 分布的均值

**返回**

Tensor，均值

variance
'''''''''

ContinuousBernoulli 分布的方差

**返回**

Tensor，方差

方法
:::::::::

prob(value)
'''''''''

计算 :attr:`value` 的概率。

**参数**

    - **value** (Tensor) - 待计算值。

**返回**

Tensor，:attr:`value` 的概率。数据类型与 `self.probs` 相同。


log_prob(value)
'''''''''

计算 value 的对数概率。

**参数**

    - **value** (Tensor) - 待计算值。

**返回**

Tensor，:attr:`value` 的对数概率。数据类型与 `self.probs` 相同。


cdf(value)
'''''''''

计算 :attr:`value` 的累计分布 quantile 值。

.. math::

    {   P(X \le t; \lambda) =
        F(t;\lambda) =
        \left\{
        \begin{aligned}
        &t & \text{ if $\lambda = \frac{1}{2}$} \\
        &\frac{\lambda^t (1 - \lambda)^{1 - t} + \lambda - 1}{2\lambda - 1} & \text{ otherwise}
        \end{aligned}
        \right. }

**参数**

    - **value** (Tensor) - 待计算值。

**返回**

Tensor: :attr:`value` 的累积分布函数对应的 quantile 值。数据类型与 `self.probs` 相同。


icdf(value)
'''''''''

计算 value 的逆累计分布值。

.. math::

    {   F^{-1}(x;\lambda) =
        \left\{
        \begin{aligned}
        &x & \text{ if $\lambda = \frac{1}{2}$} \\
        &\frac{\log(1+(\frac{2\lambda - 1}{1 - \lambda})x)}{\log(\frac{\lambda}{1-\lambda})} & \text{ otherwise}
        \end{aligned}
        \right. }

**参数**

    - **value** (Tensor) - 待计算 quantile。

**返回**

Tensor，ContinuousBernoulli 随机变量在对应 quantile 下的值。数据类型与 `self.probs` 相同。


sample(shape=())
'''''''''

从 ContinuousBernoulli 分布中生成满足特定形状的样本数据。最终生成样本形状为 ``shape+batch_shape`` 。

**参数**

    - **shape** (Sequence[int]，可选)：采样次数。

**返回**

Tensor，样本数据。其维度为 :math:`\text{sample shape} + \text{batch shape}` 。


rsample(shape=())
'''''''''

重参数化采样，生成指定维度的样本。最终生成样本形状为 ``shape+batch_shape`` 。

**参数**

    - **shape** (Sequence[int]，可选)：采样次数。

**返回**

Tensor：样本数据。其维度为 :math:`\text{sample shape} + \text{batch shape}` 。


entropy()
'''''''''

计算 ContinuousBernoulli 分布的信息熵。

.. math::

    \mathcal{H}(X) = -\log C + \left[ \log (1 - \lambda) -\log \lambda \right] \mathbb{E}(X)  - \log(1 - \lambda)

**返回**

连续伯努利分布的信息熵。


kl_divergence(other)
'''''''''

相对于另一个连续伯努利分布的 KL 散度，两个分布需要有相同的 :math:`\text{batch shape}`。

.. math::

    KL\_divergence(\lambda_1, \lambda_2) = - H - \{\log C_2 + [\log \lambda_2 -  \log (1-\lambda_2)]  \mathbb{E}_1(X) +  \log (1-\lambda_2)  \}

**参数**

    - **other** (ContinuousBernoulli) - 输入的另一个连续伯努利分布。

**返回**

相对于另一个连续伯努利分布的 KL 散度。
