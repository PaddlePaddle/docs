.. _cn_api_paddle_distribution_Normal:

Normal
-------------------------------

.. py:class:: paddle.distribution.Normal(loc, scale, name=None)


正态分布

若 `loc` 是实数，概率密度函数为：

.. math::

    pdf(x; \mu, \sigma) = \frac{1}{Z}e^{\frac {-0.5 (x - \mu)^2}  {\sigma^2} }

    Z = (2 \pi \sigma^2)^{0.5}

若 `loc` 是复数，概率密度函数为：

.. math::

    pdf(x; \mu, \sigma) = \frac{1}{Z}e^{\frac {-(x - \mu)^2}  {\sigma^2} }

    Z = \pi \sigma^2

上面的数学公式中：

- :math:`loc = \mu`：平均值；
- :math:`scale = \sigma`：标准差；
- :math:`Z`：正态分布常量。

参数
::::::::::::

    - **loc** (int|float|complex|list|tuple|numpy.ndarray|Tensor) - 正态分布平均值。数据类型为 float32、float64、complex64 或 complex128。
    - **scale** (int|float|list|tuple|numpy.ndarray|Tensor) - 正态分布标准差。数据类型为 float32 或 float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
::::::::::::


COPY-FROM: paddle.distribution.Normal


属性
:::::::::

mean
'''''''''

正态分布的均值

variance
'''''''''

正态分布的方差


方法
:::::::::

sample(shape=[], seed=0)
'''''''''

生成指定维度的样本。

**参数**

    - **shape** (Sequence[int], 可选) - 指定生成样本的维度。
    - **seed** (int) - 长整型数。

**返回**

Tensor，预先设计好维度的 Tensor，数据类型为 float32。

rsample(shape=[])
'''''''''

重参数化采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int], 可选) - 指定生成样本的维度。

**返回**

Tensor，预先设计好维度的 Tensor，数据类型为 float32。

entropy()
'''''''''

信息熵

实高斯分布信息熵的数学公式：

.. math::

    entropy(\sigma) = 0.5 \log (2 \pi e \sigma^2)

复高斯分布信息熵的数学公式：

.. math::

    entropy(\sigma) = \log (\pi e \sigma^2) + 1

上面的数学公式中：

:math:`scale = \sigma`：标准差。

**返回**

Tensor，正态分布的信息熵，数据类型为 float32。

log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。数据类型为 float32 或 float64。

**返回**

Tensor，对数概率，数据类型与 :attr:`value` 相同。

probs(value)
'''''''''

概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。数据类型为 float32 或 float64。

**返回**

Tensor，概率，数据类型与 :attr:`value` 相同。

kl_divergence(other)
'''''''''

两个正态分布之间的 KL 散度。

实高斯分布 KL 散度的数学公式：

.. math::

    KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio})

    ratio = \frac{\sigma_0}{\sigma_1}

    diff = \mu_1 - \mu_0

复高斯分布 KL 散度的数学公式：

.. math::

    KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio}

    ratio = \frac{\sigma_0}{\sigma_1}

    diff = \mu_1 - \mu_0

上面的数学公式中：

- :math:`loc = \mu_0`：当前正态分布的平均值；
- :math:`scale = \sigma_0`：当前正态分布的标准差；
- :math:`loc = \mu_1`：另一个正态分布的平均值；
- :math:`scale = \sigma_1`：另一个正态分布的标准差；
- :math:`ratio`：两个标准差之间的比例；
- :math:`diff`：两个平均值之间的差值。

**参数**

    - **other** (Normal) - Normal 的实例。

**返回**

Tensor，两个正态分布之间的 KL 散度，数据类型为 float32。
