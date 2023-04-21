.. _cn_api_distribution_Geometric:

Geometric
-------------------------------

.. py:class:: paddle.distribution.Geometric(probs)
几何分布

数学公式：

.. math::
    $P(X=k) = (1-p)^{k-1}p$

上面数学公式中：

:math:`p`：表示成功的概率。

:math:`X`：表示进行了多少次试验才获得第一次成功。

:math:`k`：表示实验次数，是一个正整数


参数
::::::::::::

    - **probs** (float|Tensor) - 几何分布成功概率参数。数据类型为float、Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.Geometric

属性
:::::::::

mean
'''''''''

均值

数学公式：

.. math::
    mean = \frac{1}{p}$

上面数学公式中：

:math:`p`：试验成功的概率。

variance
'''''''''

方差

数学公式：

.. math::
    variance = \frac{1-p}{p^2}$

上面数学公式中：

:math:`p`：试验成功的概率。

stddev
'''''''''

标准差

数学公式：

.. math::
    stddev = $\sqrt{variance} = \sqrt{\frac{1-p}{p^2}} = \frac{\sqrt{1-p}}{p}$

上面数学公式中：

:math:`p`：试验成功的概率。


方法
:::::::::

pmf(k)
'''''''''
几何分布的概率质量函数。

**参数**

    - **k** (int) - 几何分布的随机变量。

数学公式：

.. math::
    pmf(X=k) = (1-p)^{k-1} p, \quad k=1,2,3,\ldots

上面数学公式中：

:math:`p`：试验成功的概率。

:math:`k`：几何分布的随机变量。

**返回**

    - **Tensor** - value 第一次成功所需的试验次数k的概率。

log_pmf(k)
'''''''''
几何分布的对数概率质量函数。

**参数**

    - **k** (int) - 几何分布的随机变量。

数学公式：

.. math::

    \log pmf(X = k) = \log(1-p)^k p

上面数学公式中：

:math:`p`：试验成功的概率。

:math:`k`：几何分布的实验次数。

**返回**

    - **Tensor** - value 第一次成功所需的试验次数k的概率的对数。

cdf(k)
'''''''''
几何分布的累积分布函数

**参数**

    - **k** (int) - 几何分布的随机变量。

数学公式：

.. math::

    cdf(X \leq k) = 1 - (1-p)^k, \quad k=0,1,2,\ldots

上面的数学公式中：

:math:`p`：试验成功的概率。

:math:`k`：几何分布的随机变量。

**返回**

    - Tensor: value 随机变量X小于或等于某个值x的概率。

entropy()
'''''''''
几何分布的信息熵。

数学公式：

.. math::

    entropy() = -\left[\frac{1}{p} \log p + \frac{1-p}{p^2} \log (1-p) \right]

上面数学公式中：

:math:`p`：试验成功的概率。

kl_divergence(other)
'''''''''
两个 Geometric 分布之间的 KL 散度。

**参数**

    - **other** (Geometric) - Geometric 的实例。

数学公式：

.. math::
        KL(P \| Q) = \frac{p}{q} \log \frac{p}{q} + \log (1-p) - \log (1-q)

上面的数学公式中：

:math:`P`：Geometric 几何分布实例。

:math:`Q`：Geometric 几何分布实例。

:math:`p`：Geometric_p 分布试验成功的概率。

:math:`q`：Geometric_q 分布试验成功的概率。

**返回**

    - Tensor: 两个几何分布之间的 KL 散度。

sample(shape)
'''''''''
随机采样，生成指定维度的样本。

**参数**

    - **shape** (tuple(int)) - 采样的样本维度。

**返回**

    - **Tensor** - 预先设计好维度的样本数据。


rsample(shape)
'''''''''
重参数化采样。

**参数**

    - **shape** (tuple(int)) - 重参数化采样的样本维度。

**返回**

    - **Tensor** - 预先设计好维度的样本数据。
