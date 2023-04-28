.. _cn_api_distribution_Cauchy:

Cauchy
-------------------------------

.. py:class:: paddle.distribution.Cauchy(loc, scale, name=None)

柯西分布也叫柯西-洛伦兹分布，它是以奥古斯丁·路易·柯西与亨德里克·洛伦兹名字命名的连续概率分布。其在自然科学中有着非常广泛的应用。

柯西分布的概率密度函数（PDF）:

.. math::

    { f(x; loc, scale) = \frac{1}{\pi scale \left[1 + \left(\frac{x - loc}{ scale}\right)^2\right]} = { 1 \over \pi } \left[ {  scale \over (x - loc)^2 +  scale^2 } \right], }

参数
::::::::::::

    - **loc** (float|Tensor) - 定义分布峰值位置的位置参数。数据类型为 float32 或 float64。
    - **scale** (float|Tensor) - 最大值一半处的一半宽度的尺度参数。数据类型为 float32 或 float64。必须为正值。
    - **name** (str，可选) - 操作的名称，一般无需设置，默认值为 None，具体用法请参见 :ref:`api_guide_Name`。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.Cauchy

属性
:::::::::

mean
'''''''''

柯西分布的均值

**返回**

ValueError，柯西分布没有均值

variance
'''''''''

柯西分布的方差

**返回**

ValueError，柯西分布没有方差

stddev
'''''''''

柯西分布的标准差

**返回**

ValueError，柯西分布没有标准差

方法
:::::::::

sample(shape, name=None)
'''''''''

生成指定维度的样本。

.. note::
    `sample` 方法没有梯度，如果需要的话，请使用 `rsample` 方法代替。

**参数**

    - **shape** (Sequence[int]) - 指定生成样本的维度。
    - **name** (str，可选) - 操作的名称，一般无需设置，默认值为 None，具体用法请参见 :ref:`api_guide_Name`。

**返回**

Tensor，样本，其维度为 :math:`\text{sample shape} + \text{batch shape} + \text{event shape}`。

**代码示例**

COPY-FROM: paddle.distribution.Cauchy.sample

rsample(shape, name=None)
'''''''''

重参数化采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int]) - 指定生成样本的维度。
    - **name** (str，可选) - 操作的名称，一般无需设置，默认值为 None，具体用法请参见 :ref:`api_guide_Name`。

**返回**

Tensor，样本，其维度为 :math:`\text{sample shape} + \text{batch shape} + \text{event shape}`。

**代码示例**

COPY-FROM: paddle.distribution.Cauchy.rsample

prob(value)
'''''''''

``value`` 的概率密度函数。

.. math::

    { f(x; loc, scale) = \frac{1}{\pi scale \left[1 + \left(\frac{x - loc}{ scale}\right)^2\right]} = { 1 \over \pi } \left[ {  scale \over (x - loc)^2 +  scale^2 } \right], }

**参数**

    - **value** (Tensor) - 输入 Tensor。

**返回**

Tensor， ``value`` 的概率密度函数。

**代码示例**

COPY-FROM: paddle.distribution.Cauchy.prob

log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

**返回**

Tensor， ``value`` 的对数概率密度函数。

**代码示例**

COPY-FROM: paddle.distribution.Cauchy.log_prob

cdf(value)
'''''''''

``value`` 的累积分布函数 （CDF）

.. math::

    { \frac{1}{\pi} \arctan\left(\frac{x-loc}{ scale}\right)+\frac{1}{2}\! }

**参数**

    - **value** (Tensor) - 输入 Tensor。

**返回**

Tensor， ``value`` 的累积分布函数。

**代码示例**

COPY-FROM: paddle.distribution.Cauchy.cdf

entropy()
'''''''''

柯西分布的信息熵。

.. math::

    { \log(4\pi scale)\! }

**返回**

Tensor，柯西分布的信息熵。

**代码示例**

COPY-FROM: paddle.distribution.Cauchy.entropy

kl_divergence(other)
'''''''''

两个柯西分布之间的 KL 散度。

.. note::
    [1] Frédéric Chyzak, Frank Nielsen, A closed-form formula for the Kullback-Leibler divergence between Cauchy distributions, 2019

**参数**

    - **other** (Cauchy) - ``Cauchy`` 的实例。

**返回**

Tensor，两个柯西分布之间的 KL 散度。

**代码示例**

COPY-FROM: paddle.distribution.Cauchy.kl_divergence
