.. _cn_api_distribution_Normal:

LogNormal
-------------------------------

.. py:class:: paddle.distribution.LogNormal(loc, scale, name=None)


对数正态分布

.. math::

    X \sim Normal(\mu, \sigma)

    Y = exp(X) \sim LogNormal(\mu, \sigma)


由于对数正态分布是基于正态分布的变换得到的分布，一般称 :math:`Normal(\mu, \sigma)` 是 :math:`LogNormal(\mu, \sigma)` 的基础分布。

数学公式：

概率密度函数

.. math::

    pdf(x; \mu, \sigma) = \frac{1}{\sigma x \sqrt{2\pi}}e^{(-\frac{(ln(x) - \mu)^2}{2\sigma^2})}

上面的数学公式中：

- :math:`loc = \mu`：基础正态分布的平均值；
- :math:`scale = \sigma`：基础正态分布的标准差。

参数
::::::::::::

    - **loc** (int|float|list|tuple|numpy.ndarray|Tensor) - 基础正态分布的平均值。
    - **scale** (int|float|list|tuple|numpy.ndarray|Tensor) - 基础正态分布的标准差。

代码示例
::::::::::::


COPY-FROM: paddle.distribution.LogNormal


属性
:::::::::

mean
'''''''''

对数正态分布的均值

variance
'''''''''

对数正态分布的方差


方法
:::::::::

sample(shape=(), seed=0)
'''''''''

生成指定维度的样本。

**参数**

    - **shape** (Sequence[int], 可选) - 指定生成样本的维度。
    - **seed** (int) - 长整型数。

**返回**

Tensor，预先设计好维度的样本数据。

rsample(shape=())
'''''''''

重参数化采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int], 可选) - 指定生成样本的维度。

**返回**

Tensor，预先设计好维度的样本数据。

entropy()
'''''''''

信息熵

数学公式：

.. math::

    entropy(\sigma) = 0.5 \log (2 \pi e \sigma^2) + \mu

上面的数学公式中：

- :math:`loc = \mu`：基础正态分布的平均值；
- :math:`scale = \sigma`：基础正态分布的标准差。

**返回**

Tensor，对数正态分布的信息熵。

log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

**返回**

Tensor，对数概率，数据类型与 :attr:`value` 相同。

probs(value)
'''''''''

概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

**返回**

Tensor，概率，数据类型与 :attr:`value` 相同。

kl_divergence(other)
'''''''''

两个对数正态分布之间的 KL 散度。

数学公式：

.. math::

    KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio})

    ratio = \frac{\sigma_0}{\sigma_1}

    diff = \mu_1 - \mu_0

上面的数学公式中：

- :math:`loc = \mu_0`：当前对数分布对应的基础分布的平均值；
- :math:`scale = \sigma_0`：当前对数分布对应的基础分布的标准差；
- :math:`loc = \mu_1`：另一个对数分布对应的基础分布的平均值；
- :math:`scale = \sigma_1`：另一个对数分布对应的基础分布的标准差；
- :math:`ratio`：两个标准差之间的比例；
- :math:`diff`：两个平均值之间的差值。

**参数**

    - **other** (LogNormal) - LogNormal 的实例。

**返回**

Tensor，两个对数正态分布之间的 KL 散度。
