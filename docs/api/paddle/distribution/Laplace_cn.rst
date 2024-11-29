.. _cn_api_paddle_distribution_Laplace:

Laplace
-------------------------------

.. py:class:: paddle.distribution.Laplace(loc, scale)
拉普拉斯分布

数学公式：

.. math::
    pdf(x; \mu, \sigma) = \frac{1}{2 * \sigma} * e^{\frac {-|x - \mu|}{\sigma}}

上面的数学公式中：

:math:`loc = \mu`：拉普拉斯分布位置参数。

:math:`scale = \sigma`：拉普拉斯分布尺度参数。


参数
::::::::::::

    - **loc** (int|float|Tensor) - 拉普拉斯分布位置参数。数据类型为 int、float 或 Tensor。
    - **scale** (int|float|Tensor) - 拉普拉斯分布尺度参数。数据类型为 int、float 或 Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.Laplace

属性
:::::::::

mean
'''''''''

均值

variance
'''''''''

方差

数学公式：

.. math::
    variance = 2 * \sigma^2

上面的数学公式中：

:math:`scale = \sigma`：拉普拉斯分布尺度参数。

stddev
'''''''''

标准差

数学公式：

.. math::
    stddev = \sqrt{2} * \sigma

上面的数学公式中：

:math:`scale = \sigma`：拉普拉斯分布尺度参数。


方法
:::::::::

cdf(value)
'''''''''
累积分布函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

数学公式：

.. math::
    cdf(value) = 0.5 - 0.5 * sign(value - \mu) * e^\frac{-|(\mu - \sigma)|}{\sigma}

上面的数学公式中：

:math:`loc = \mu`：拉普拉斯分布位置参数。

:math:`scale = \sigma`：拉普拉斯分布尺度参数。

**返回**

    - Tensor: value 对应 Laplace 累积分布函数下的值。


icdf(value)
'''''''''
逆累积分布函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

数学公式：

.. math::

    cdf^{-1}(value)= \mu - \sigma * sign(value - 0.5) * ln(1 - 2 * |value-0.5|)

上面的数学公式中：

:math:`loc = \mu`：拉普拉斯分布位置参数。

:math:`scale = \sigma`：拉普拉斯分布尺度参数。

**返回**

    - Tensor: value 对应 Laplace 逆累积分布函数下的值。


sample(shape=[])
'''''''''

生成指定维度的样本。

**参数**

    - **shape** (Sequence[int]，可选) - 1 维元组，指定生成样本的维度，默认为[]。

**返回**

    - Tensor: 预先设计好维度的样本数据。


rsample(shape=[])
'''''''''

生成指定维度的样本（重参数采样）。

**参数**

    - **shape** (Sequence[int]，可选) - 1 维元组，指定生成样本的维度，默认为[]。

**返回**

    - Tensor: 预先设计好维度的样本数据。


entropy()
'''''''''

信息熵

数学公式：

.. math::
    entropy() = 1 + log(2 * \sigma)

上面的数学公式中：

:math:`scale = \sigma`：拉普拉斯分布尺度参数.

**返回**

    - Tensor: Laplace 分布的信息熵。


log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor|Scalar) - 待计算值。

数学公式：

.. math::
    log\_prob(value) = \frac{-log(2 * \sigma) - |value - \mu|}{\sigma}

上面的数学公式中：

:math:`loc = \mu`：拉普拉斯分布位置参数。

:math:`scale = \sigma`：拉普拉斯分布尺度参数.

**返回**

    - Tensor: value 的对数概率。


prob(value)
'''''''''

概率密度函数

**参数**

    - **value** (Tensor|Scalar) - 待计算值。

数学公式：

.. math::
    prob(value) = e^{\frac{-log(2 * \sigma) - |value - \mu|}{\sigma}}

上面的数学公式中：

:math:`loc = \mu`：拉普拉斯分布位置参数。

:math:`scale = \sigma`：拉普拉斯分布尺度参数.

**返回**

    - Tensor: value 的概率。


kl_divergence(other)
'''''''''

两个 Laplace 分布之间的 KL 散度。


**参数**

    - **other** (Laplace) - Laplace 的实例。

数学公式：

.. math::
    KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio})

.. math::
    ratio = \frac{\sigma_0}{\sigma_1}

.. math::
    diff = \mu_1 - \mu_0

上面的数学公式中：

:math:`loc = \mu_0`：当前拉普拉斯分布的位置参数。

:math:`scale = \sigma_0`：当前拉普拉斯分布的尺度参数。

:math:`loc = \mu_1`：另一个拉普拉斯分布的位置参数。

:math:`scale = \sigma_1`：另一个拉普拉斯分布的尺度参数.

:math:`ratio`：两个尺度参数之间的比例。

:math:`diff`：两个位置参数之间的差值。

**返回**

    - Tensor: 两个拉普拉斯分布之间的 KL 散度。
