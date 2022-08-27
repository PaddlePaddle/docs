.. _cn_api_distribution_Laplace:

Laplace
-------------------------------

.. py:class:: paddle.distribution.Laplace(loc, scale, name=None)


拉普拉斯分布

数学公式：

.. math::


    pdf(x; \mu, \sigma) = \frac{1}{Z}e^ \frac {-|x - \mu|}  \sigma
    Z = 2 \sigma

上面的数学公式中：

:math:`loc = \mu`：拉普拉斯分布平均值。
:math:`scale = \sigma`：拉普拉斯分布标准差。
:math:`Z`：拉普拉斯分布常量。

参数
::::::::::::

    - **loc** (int|float|numpy.ndarray|Tensor) - 拉普拉斯分布平均值。数据类型为 int、float、numpy.ndarray 或 Tensor。
    - **scale** (int|float|numpy.ndarray|Tensor) - 拉普拉斯分布标准差。数据类型为 int、float、numpy.ndarray 或 Tensor。

代码示例
::::::::::::


COPY-FROM: paddle.distribution.Laplace

方法
:::::::::

cdf
'''''''''
累积分布函数

cdf(x) = 0.5 - 0.5 * sign(\mu - \sigma) * e^\frac{|-(\mu - \sigma)|} {\sigma}

上面的数学公式中：
:math:`loc = \mu`：拉普拉斯分布平均值。
:math:`scale = \sigma`：拉普拉斯分布标准差。

icdf
'''''''''
逆累积分布函数

cdf^{-1}(p)= \mu -\sigma sign(p-0.5)ln(1-2|p-0.5|)

上面的数学公式中：
:math:`loc = \mu`：拉普拉斯分布平均值。
:math:`scale = \sigma`：拉普拉斯分布标准差。
sample(shape)
'''''''''

生成指定维度的样本。

**参数**

- **shape** (tuple) - 1 维元组，指定生成样本的维度，默认为()。

**返回**

- Tensor: 样本数据。

entropy()
'''''''''

信息熵

数学公式：

.. math::

    entropy(\sigma) = 1+log(2*\sigma)

上面的数学公式中：

:math:`scale = \sigma`：标准差。

**返回**

- Tensor: 信息熵。

log_prob(value)
'''''''''

对数概率密度函数

**参数**

- **value** (Tensor|Scalar) - 待计算值。

**返回**

- Tensor: 对数概率。

probs(value)
'''''''''

概率密度函数

**参数**

- **value** (Tensor|Scalar) - 输入张量。

**返回**

Tensor，概率，数据类型与 value 相同。

kl_divergence(other)
'''''''''

两个 Laplace 分布之间的 KL 散度。

数学公式：

.. math::

    KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio})

    ratio = \frac{\sigma_0}{\sigma_1}

    diff = \mu_1 - \mu_0

上面的数学公式中：

:math:`loc = \mu_0`：当前拉普拉斯分布的平均值。
:math:`scale = \sigma_0`：当前拉普拉斯分布的标准差。
:math:`loc = \mu_1`：另一个拉普拉斯分布的平均值。
:math:`scale = \sigma_1`：另一个拉普拉斯分布的标准差。
:math:`ratio`：两个标准差之间的比例。
:math:`diff`：两个平均值之间的差值。

**参数**

- **other** (Laplace) - Laplace 的实例。

**返回**

- Tensor: 两个拉普拉斯分布之间的 KL 散度。
