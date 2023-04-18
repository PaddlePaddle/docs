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

.. code-block:: python

    import paddle
    from paddle.distribution import Cauchy

    # init Cauchy with float
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.entropy())
    # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        2.71334577)

    # init Cauchy with N-Dim tensor
    rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.entropy())
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [2.53102422, 3.22417140])

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

.. code-block:: python

    import paddle
    from paddle.distribution import Cauchy

    # init Cauchy with float
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.sample([10]).shape)
    # [10]

    # init Cauchy with 0-Dim tensor
    rv = Cauchy(loc=paddle.full((), 0.1), scale=paddle.full((), 1.2))
    print(rv.sample([10]).shape)
    # [10]

    # init Cauchy with N-Dim tensor
    rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.sample([10]).shape)
    # [10, 2]

    # sample 2-Dim data
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.sample([10, 2]).shape)
    # [10, 2]

    rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.sample([10, 2]).shape)
    # [10, 2, 2]

rsample(shape, name=None)
'''''''''

重参数化采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int]) - 指定生成样本的维度。
    - **name** (str，可选) - 操作的名称，一般无需设置，默认值为 None，具体用法请参见 :ref:`api_guide_Name`。

**返回**

Tensor，样本，其维度为 :math:`\text{sample shape} + \text{batch shape} + \text{event shape}`。

**代码示例**

.. code-block:: python

    import paddle
    from paddle.distribution import Cauchy

    # init Cauchy with float
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.rsample([10]).shape)
    # [10]

    # init Cauchy with 0-Dim tensor
    rv = Cauchy(loc=paddle.full((), 0.1), scale=paddle.full((), 1.2))
    print(rv.rsample([10]).shape)
    # [10]

    # init Cauchy with N-Dim tensor
    rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.rsample([10]).shape)
    # [10, 2]

    # sample 2-Dim data
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.rsample([10, 2]).shape)
    # [10, 2]

    rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.rsample([10, 2]).shape)
    # [10, 2, 2]

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

.. code-block:: python

    import paddle
    from paddle.distribution import Cauchy

    # init Cauchy with float
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.prob(paddle.to_tensor(1.5)))
    # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [0.11234467])

    # broadcast to value
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.prob(paddle.to_tensor([1.5, 5.1])))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [0.11234467, 0.01444674])

    # init Cauchy with N-Dim tensor
    rv = Cauchy(loc=paddle.to_tensor([0.1, 0.1]), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.prob(paddle.to_tensor([1.5, 5.1])))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [0.10753712, 0.02195240])

    # init Cauchy with N-Dim tensor with broadcast
    rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.prob(paddle.to_tensor([1.5, 5.1])))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [0.10753712, 0.02195240])

log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。

**返回**

Tensor， ``value`` 的对数概率密度函数。

**代码示例**

.. code-block:: python

    import paddle
    from paddle.distribution import Cauchy

    # init Cauchy with float
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.log_prob(paddle.to_tensor(1.5)))
    # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [-2.18618369])

    # broadcast to value
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.log_prob(paddle.to_tensor([1.5, 5.1])))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [-2.18618369, -4.23728657])

    # init Cauchy with N-Dim tensor
    rv = Cauchy(loc=paddle.to_tensor([0.1, 0.1]), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.log_prob(paddle.to_tensor([1.5, 5.1])))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [-2.22991920, -3.81887865])

    # init Cauchy with N-Dim tensor with broadcast
    rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.log_prob(paddle.to_tensor([1.5, 5.1])))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [-2.22991920, -3.81887865])

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

.. code-block:: python

    import paddle
    from paddle.distribution import Cauchy

    # init Cauchy with float
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.cdf(paddle.to_tensor(1.5)))
    # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [0.77443725])

    # broadcast to value
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.cdf(paddle.to_tensor([1.5, 5.1])))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [0.77443725, 0.92502367])

    # init Cauchy with N-Dim tensor
    rv = Cauchy(loc=paddle.to_tensor([0.1, 0.1]), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.cdf(paddle.to_tensor([1.5, 5.1])))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [0.80256844, 0.87888104])

    # init Cauchy with N-Dim tensor with broadcast
    rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.cdf(paddle.to_tensor([1.5, 5.1])))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [0.80256844, 0.87888104])

entropy()
'''''''''

柯西分布的信息熵。

.. math::

    { \log(4\pi scale)\! }

**返回**

Tensor，柯西分布的信息熵。

**代码示例**

.. code-block:: python

    import paddle
    from paddle.distribution import Cauchy

    # init Cauchy with float
    rv = Cauchy(loc=0.1, scale=1.2)
    print(rv.entropy())
    # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        2.71334577)

    # init Cauchy with N-Dim tensor
    rv = Cauchy(loc=paddle.to_tensor(0.1), scale=paddle.to_tensor([1.0, 2.0]))
    print(rv.entropy())
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [2.53102422, 3.22417140])

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

.. code-block:: python

    import paddle
    from paddle.distribution import Cauchy

    rv = Cauchy(loc=0.1, scale=1.2)
    rv_other = Cauchy(loc=paddle.to_tensor(1.2), scale=paddle.to_tensor([2.3, 3.4]))
    print(rv.kl_divergence(rv_other))
    # Tensor(shape=[2], dtype=float32, place=Place(cpu), stop_gradient=True,
    #        [0.19819736, 0.31532931])
