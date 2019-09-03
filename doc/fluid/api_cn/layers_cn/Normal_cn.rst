.. _cn_api_fluid_layers_Normal:

Normal
-------------------------------

.. py:class:: paddle.fluid.layers.Normal(loc, scale)

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
    - **loc** (float|list|numpy.ndarray|Variable) - 正态分布平均值。
    - **scale** (float|list|numpy.ndarray|Variable) - 正态分布标准差。

**代码示例**：

.. code-block:: python

    from paddle.fluid import layers
	from paddle.fluid.layers import Normal

    # 定义单个数的正态分布。
    dist = Normal(loc=0., scale=3.)
    # 定义一组有两个数的正态分布。
    # 第一组为均值1，标准差11，第二组为均值2，标准差22。
    dist = Normal(loc=[1, 2.], scale=[11, 22.])
    # 得到3个样本, 返回一个 3 x 2 向量。
    dist.sample([3])

    # 定义一个为两个参数的正态分布。
    # 均值都是1，标准差不同。
    dist = Normal(loc=1., scale=[11, 22.])

    # 输入变量
    dims = 3

    loc = layers.data(name='loc', shape=[dims], dtype='float32')
    scale = layers.data(name='scale', shape=[dims], dtype='float32')
    other_loc = layers.data(
        name='other_loc', shape=[dims], dtype='float32')
    other_scale = layers.data(
        name='other_scale', shape=[dims], dtype='float32')
    values = layers.data(name='values', shape=[dims], dtype='float32')

    normal = Normal(loc, scale)
    other_normal = Normal(other_loc, other_scale)

    sample = normal.sample([2, 3])
    entropy = normal.entropy()
    lp = normal.log_prob(values)
    kl = normal.kl_divergence(other_normal)


.. py:function:: sample(shape, seed=0)

生成指定形状的样本

参数：
    - **shape** (list) - int32的1维列表，指定生成样本的shape。
    - **seed** (int) - 长整型数
    
返回：预备好维度shape的向量

返回类型：变量（Variable）

.. py:function:: entropy()

信息熵
    
返回：正态分布的信息熵

返回类型：变量（Variable）

.. py:function:: log_prob(value)

Log概率密度函数

参数：
    - **value** (Variable) - 输入向量。
    
返回：log概率

返回类型：变量（Variable）

.. py:function:: kl_divergence(other)

两个正态分布之间的KL-divergence。

参数：
    - **other** (Normal) - Normal实例。
    
返回：两个正态分布之间的KL-divergence

返回类型：变量（Variable）






