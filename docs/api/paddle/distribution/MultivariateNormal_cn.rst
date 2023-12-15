.. _cn_api_distribution_MultivariateNormal:

MultivariateNormal
-------------------------------

.. py:class:: paddle.distribution.MultivariateNormal(loc, covariance_matrix)
多元正态分布

数学公式：

.. math::
    f_\boldsymbol{X}(x_1,...,x_k) = \frac{exp(-\frac{1}{2}$\mathbf{(\boldsymbol{x - \mu})}^\top$\boldsymbol{\Sigma}^{-1}(\boldsymbol{x - \mu}))}{\sqrt{(2\pi)^k\left| \boldsymbol{\Sigma} \right|}}

上面数学公式中：

:math:`loc = \boldsymbol{\mu}`：多元正态分布位置参数。

:math:`covariance\_matrix = \boldsymbol{\Sigma}`：多元正态分布协方差矩阵，且协方差矩阵为半正定矩阵时成立。


参数
::::::::::::

    - **loc** (Tensor) -多元正态分布位置参数。数据类型为 Tensor。
    - **covariance_matrix** (Tensor) - 多元正态分布协方差矩阵参数。数据类型为 Tensor，且该参数必须为半正定矩阵。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.MultivariateNormal

属性
:::::::::

mean
'''''''''

均值

数学公式：

.. math::
    mean = \boldsymbol{\mu}

上面数学公式中：

:math:`loc = \boldsymbol{\mu}`：多元正态分布位置参数。

variance
'''''''''

方差

数学公式：

.. math::
    variance = \boldsymbol{\sigma^2}

上面数学公式中：

:math:`scale = \boldsymbol{\sigma}`：多元正态分布协方差矩阵经过矩阵分解后得到的尺度向量。


stddev
'''''''''

标准差

数学公式：

.. math::
    stddev = \boldsymbol{\sigma}

上面数学公式中：

:math:`scale = \boldsymbol{\sigma}`：多元正态分布协方差矩阵经过矩阵分解后得到的尺度向量。

方法
:::::::::

prob(value)
'''''''''

多元正态分布的概率密度函数。

**参数**

    - **value** (Tensor) - 待计算的值。

数学公式：

.. math::
    prob(value) = \frac{exp(-\frac{1}{2}$\mathbf{(\boldsymbol{value - \mu})}^\top$\boldsymbol{\Sigma}^{-1}(\boldsymbol{value- \mu}))}{\sqrt{(2\pi)^k\left| \boldsymbol{\Sigma} \right|}}

上面数学公式中：

:math:`loc = \boldsymbol{\mu}`：多元正态分布位置参数。

:math:`covariance\_matrix = \boldsymbol{\Sigma}`：多元正态分布协方差矩阵，且协方差矩阵为半正定矩阵时成立。


**返回**

    - **Tensor** - 在多元正态分布下的概率值。

log_prob(value)
'''''''''
多元正态分布的对数概率密度函数。

**参数**

    - **value** (Tensor) - 待计算的值。

数学公式：

.. math::

    log\_prob(value) = log(\frac{exp(-\frac{1}{2}$\mathbf{(\boldsymbol{value - \mu})}^\top$\boldsymbol{\Sigma}^{-1}(\boldsymbol{value- \mu}))}{\sqrt{(2\pi)^k\left| \boldsymbol{\Sigma} \right|}})

上面数学公式中：

:math:`loc = \boldsymbol{\mu}`：多元正态分布位置参数。

:math:`covariance\_matrix = \boldsymbol{\Sigma}`：多元正态分布协方差矩阵，且协方差矩阵为半正定矩阵时成立。


**返回**

    - **Tensor** - 在多元正态分布下的概率值。

entropy(scale)
'''''''''
多元正态分布的信息熵。

数学公式：

.. math::

    entropy() = \frac{k}{2}(\ln 2\pi + 1) + \frac{1}{2}\ln \left| \boldsymbol{\Sigma} \right|

上面数学公式中：

:math:`k`：多元正太分布向量的维度，比如一维向量 k=1，二维向量（矩阵） k=2。

:math:`covariance\_matrix = \boldsymbol{\Sigma}`：多元正态分布协方差矩阵，且协方差矩阵为半正定矩阵时成立。

sample(shape)
'''''''''
随机采样，生成指定维度的样本。

**参数**

    - **shape** (list[int]) - 1 维列表，指定样本的维度。

**返回**

    - **Tensor** - 预先设计好维度的样本数据。


rsample(shape)
'''''''''
重参数化采样。

**参数**

    - **shape** (list[int]) - 1 维列表，指定样本的维度。

**返回**

    - **Tensor** - 预先设计好维度的样本数据。

kl_divergence(other)
'''''''''

两个 MultivariateNormal 分布之间的 KL 散度。


**参数**

    - **other** (MultivariateNormal) - MultivariateNormal 的实例。

数学公式：

.. math::
    KL\_divergence(\boldsymbol{\mu_1}, \boldsymbol{\Sigma_1}; \boldsymbol{\mu_2}, \boldsymbol{\Sigma_2}) = \frac{1}{2}\Big \{\log ratio -n + tr(\boldsymbol{\Sigma_2}^{-1}\boldsymbol{\Sigma_1}) + $\mathbf{(diff)}^\top$\boldsymbol{\Sigma_2}^{-1}\boldsymbol{(diff)} \Big \}

.. math::
    ratio = \frac{\left| \boldsymbol{\Sigma_2} \right|}{\left| \boldsymbol{\Sigma_1} \right|}

.. math::
    \boldsymbol{diff} = \boldsymbol{\mu_2} - \boldsymbol{\mu_1}

上面的数学公式中：

:math:`loc = \boldsymbol{\mu_1}`：当前多元正态分布的位置参数。

:math:`covariance\_matrix = \boldsymbol{\Sigma_1}`：当前多元正态分布的协方差矩阵。

:math:`loc = \boldsymbol{\mu_2}`：另一个多元正态分布的位置参数。

:math:`covariance\_matrix = \boldsymbol{\Sigma_2}`：另一个多元正态分布的协方差矩阵。

:math:`ratio`：两个协方差矩阵的行列式值的比值。

:math:`diff`：两个位置参数之间的差值。

:math:`n`：维度。

:math:`tr`：矩阵的迹。

**返回**

    - Tensor: 两个多元正态分布之间的 KL 散度。
