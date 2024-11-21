.. _cn_api_paddle_distribution_Gumbel:

Gumbel
-------------------------------

.. py:class:: paddle.distribution.Gumbel(loc, scale)
耿贝尔分布

数学公式：

.. math::
    F(x; \mu, \beta) = e^{-e^{\frac {-(x-\mu)} {\beta}}}

上面数学公式中：

:math:`loc = \mu`：耿贝尔分布位置参数。

:math:`scale = \beta`：耿贝尔分布尺度参数。


参数
::::::::::::

    - **loc** (int|float|Tensor) - 耿贝尔分布位置参数。数据类型为 int、float、Tensor。
    - **scale** (int|float|Tensor) - 耿贝尔分布尺度参数。数据类型为 int、float、Tensor。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.Gumbel

属性
:::::::::

mean
'''''''''

均值

数学公式：

.. math::
    mean = -\gamma

上面数学公式中：

:math:`\gamma`：欧拉常数。

variance
'''''''''

方差

数学公式：

.. math::
    variance = \frac{1}{6}{\pi^2\beta^2}

上面数学公式中：

:math:`scale = \beta`：耿贝尔分布尺度参数。

stddev
'''''''''

标准差

数学公式：

.. math::
    stddev = \frac{1}{\sqrt{6}} {\pi\beta}

上面数学公式中：

:math:`scale = \beta`：耿贝尔分布尺度参数。


方法
:::::::::

prob(value)
'''''''''
耿贝尔分布的概率密度函数。

**参数**

    - **value** (Tensor|Scalar) - 待计算的值。

数学公式：

.. math::
    prob(value) = e^{-e^{\frac {-(value-\mu)} {\beta}}}

上面数学公式中：

:math:`loc = \mu`：耿贝尔分布位置参数。

:math:`scale = \beta`：耿贝尔分布尺度参数。

**返回**

    - **Tensor** - value 在耿贝尔分布下的概率值。

log_prob(value)
'''''''''
耿贝尔分布的对数概率密度函数。

**参数**

    - **value** (Tensor|Scalar) - 待计算的值。

数学公式：

.. math::

    log\_prob(value) = log(e^{-e^{\frac {-(value-\mu)} {\beta}}})

上面数学公式中：

:math:`loc = \mu`：耿贝尔分布位置参数。

:math:`scale = \beta`：耿贝尔分布尺度参数。

**返回**

    - **Tensor** - value 在耿贝尔分布下的对数概率值。

cdf(value)
'''''''''
累积分布函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

数学公式：

.. math::
    cdf(value) = e^{-e^{\frac {-(value-\mu)} {\beta}}}

上面的数学公式中：

:math:`loc = \mu`：耿贝尔分布位置参数。

:math:`scale = \beta`：耿贝尔分布尺度参数。

**返回**

    - Tensor: value 对应 Gumbel 累积分布函数下的值。

entropy(scale)
'''''''''
耿贝尔分布的信息熵。

**参数**

    - **scale** (int|float|Tensor) - 耿贝尔分布的尺度参数。

数学公式：

.. math::

    entropy(scale) = ln(\beta) + 1 + γ

上面数学公式中：

:math:`scale = \beta`：耿贝尔分布尺度参数。

:math:`\gamma`：欧拉常数。


sample(shape=[])
'''''''''
随机采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int]，可选) - 1 维列表，指定样本的维度。

**返回**

    - **Tensor** - 预先设计好维度的样本数据。


rsample(shape=[])
'''''''''
重参数化采样。

**参数**

    - **shape** (Sequence[int]，可选) - 1 维列表，指定样本的维度。

**返回**

    - **Tensor** - 预先设计好维度的样本数据。
