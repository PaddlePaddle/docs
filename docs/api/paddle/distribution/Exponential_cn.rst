.. _cn_api_paddle_distribution_Exponential:

Exponential
-------------------------------

.. py:class:: paddle.distribution.Exponential(rate)

指数分布

指数分布的概率密度满足一下公式：

.. math::

    f(x; \theta) =  \theta e^{- \theta x },  (x \ge 0) $$

上面数学公式中：

    :math:`rate=\theta`：表示率参数。


参数
::::::::::::

    - **rate** (float|Tensor) - 率参数，该值必须大于零。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.Exponential

属性
:::::::::

mean
'''''''''
指数分布的均值。


variance
'''''''''
指数分布的方差。


方法
:::::::::

prob(value)
'''''''''
指数分布的概率密度函数。

**参数**

    - **value** (float|Tensor) - 输入值。

数学公式：
.. math::

    f(x; \theta) =  \theta e^{- \theta x },  (x \ge 0) $$

上面数学公式中：

    :math:`rate=\theta`：表示率参数。

**返回**

    - **Tensor** - value 对应的概率密度。


log_prob(value)
'''''''''
指数分布的对数概率密度函数。

**参数**

    - **value** (float|Tensor) - 输入值。

**返回**

    - **Tensor** - value 对应的对数概率密度。

entropy()
'''''''''
指数分布的信息熵。

**返回**

    - Tensor: 信息熵。

cdf(k)
'''''''''
指数分布的累积分布函数。

**参数**

    - **value** (float|Tensor) - 输入值。

数学公式：

.. math::

    cdf(x; \theta) = 1 - e^{- \theta x }, (x \ge 0)

上面的数学公式中：

    :math:`rate=\theta`：表示率参数。

**返回**

    - Tensor: value 对应的累积分布。

icdf(k)
'''''''''
指数分布的逆累积分布函数。

**参数**

    - **value** (float|Tensor) - 输入值。

数学公式：

.. math::

    icdf(x; \theta) = -\frac{ 1 }{ \theta } ln(1 + x), (x \ge 0)

上面的数学公式中：

    :math:`rate=\theta`：表示率参数。

**返回**

    - Tensor: value 对应的逆累积分布。


kl_divergence(other)
'''''''''
两个指数分布之间的 KL 散度。

**参数**

    - **other** (Geometric) - Exponential 的实例。

**返回**

    - Tensor: 两个指数分布之间的 KL 散度。


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
