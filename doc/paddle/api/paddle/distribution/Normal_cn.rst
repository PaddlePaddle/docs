.. _cn_api_distribution_Normal:

Normal
-------------------------------

.. py:class:: paddle.distribution.Normal(loc, scale, name=None)




正态分布

数学公式：

.. math::

    pdf(x; \mu, \sigma) = \frac{1}{Z}e^{\frac {-0.5 (x - \mu)^2}  {\sigma^2} }

    Z = (2 \pi \sigma^2)^{0.5}

上面的数学公式中：

:math:`loc = \mu` : 平均值。
:math:`scale = \sigma` : 标准差。
:math:`Z`: 正态分布常量。

参数：
    - **loc** (int|float|list|numpy.ndarray|Tensor) - 正态分布平均值。数据类型为int、float、list、numpy.ndarray或Tensor。
    - **scale** (int|float|list|numpy.ndarray|Tensor) - 正态分布标准差。数据类型为int、float、list、numpy.ndarray或Tensor。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.distribution import Normal

    # Define a single scalar Normal distribution.
    dist = Normal(loc=0., scale=3.)
    # Define a batch of two scalar valued Normals.
    # The first has mean 1 and standard deviation 11, the second 2 and 22.
    dist = Normal(loc=[1., 2.], scale=[11., 22.])
    # Get 3 samples, returning a 3 x 2 tensor.
    dist.sample([3])

    # Define a batch of two scalar valued Normals.
    # Both have mean 1, but different standard deviations.
    dist = Normal(loc=1., scale=[11., 22.])

    # Complete example
    value_npdata = np.array([0.8], dtype="float32")
    value_tensor = paddle.to_tensor(value_npdata)

    normal_a = Normal([0.], [1.])
    normal_b = Normal([0.5], [2.])
    sample = normal_a.sample([2])
    # a random tensor created by normal distribution with shape: [2, 1]
    entropy = normal_a.entropy()
    # [1.4189385] with shape: [1]
    lp = normal_a.log_prob(value_tensor)
    # [-1.2389386] with shape: [1]
    p = normal_a.probs(value_tensor)
    # [0.28969154] with shape: [1]
    kl = normal_a.kl_divergence(normal_b)
    # [0.34939718] with shape: [1]


.. py:function:: sample(shape, seed=0)

生成指定维度的样本

参数：
    - **shape** (list) - 1维列表，指定生成样本的维度。数据类型为int32。
    - **seed** (int) - 长整型数。

返回：预先设计好维度的张量, 数据类型为float32

返回类型：Tensor

.. py:function:: entropy()

信息熵

数学公式：

.. math::

    entropy(\sigma) = 0.5 \log (2 \pi e \sigma^2)

上面的数学公式中：

:math:`scale = \sigma` : 标准差。

返回：正态分布的信息熵, 数据类型为float32

返回类型：Tensor

.. py:function:: log_prob(value)

对数概率密度函数

参数：
    - **value** (Tensor) - 输入张量。数据类型为float32或float64。

返回：对数概率, 数据类型与value相同

返回类型：Tensor

.. py:function:: probs(value)

概率密度函数

参数：
    - **value** (Tensor) - 输入张量。数据类型为float32或float64。

返回：概率, 数据类型与value相同

返回类型：Tensor

.. py:function:: kl_divergence(other)

两个正态分布之间的KL散度。

数学公式：

.. math::

    KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio})

    ratio = \frac{\sigma_0}{\sigma_1}

    diff = \mu_1 - \mu_0

上面的数学公式中：

:math:`loc = \mu_0`: 当前正态分布的平均值。
:math:`scale = \sigma_0`: 当前正态分布的标准差。
:math:`loc = \mu_1`: 另一个正态分布的平均值。
:math:`scale = \sigma_1`: 另一个正态分布的标准差。
:math:`ratio`: 两个标准差之间的比例。
:math:`diff`: 两个平均值之间的差值。

参数：
    - **other** (Normal) - Normal的实例。

返回：两个正态分布之间的KL散度, 数据类型为float32

返回类型：Tensor





