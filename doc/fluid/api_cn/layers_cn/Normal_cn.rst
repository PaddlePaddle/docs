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

    import numpy as np
    from paddle.fluid import layers
	  from paddle.fluid.layers import Normal

    # 定义参数为float的正态分布。
    dist = Normal(loc=0., scale=3.)
    # 定义一组有两个数的正态分布。
    # 第一组为均值1，标准差11，第二组为均值2，标准差22。
    dist = Normal(loc=[1., 2.], scale=[11., 22.])
    # 得到3个样本, 返回一个 3 x 2 张量。
    dist.sample([3])

    # 通过广播的方式，定义一个两个参数的正态分布。
    # 均值都是1，标准差不同。
    dist = Normal(loc=1., scale=[11., 22.])

    # 一个完整的例子
    value_npdata = np.array([0.8], dtype="float32")
    value_tensor = layers.create_tensor(dtype="float32")
    layers.assign(value_npdata, value_tensor)

    normal_a = Normal([0.], [1.])
    normal_b = Normal([0.5], [2.])

    sample = normal_a.sample([2])
    # 一个由定义好的正太分布随机生成的张量，shape为: [2, 1]
    entropy = normal_a.entropy()
    # [1.4189385] with shape: [1]
    lp = normal_a.log_prob(value_tensor)
    # [-1.2389386] with shape: [1]
    kl = normal_a.kl_divergence(normal_b)
    # [0.34939718] with shape: [1]


.. py:function:: sample(shape, seed=0)

生成指定形状的样本

参数：
    - **shape** (list) - int32的1维列表，指定生成样本的shape。
    - **seed** (int) - 长整型数。
    
返回：预先设计好形状的张量, 数据类型为float32

返回类型：Variable

.. py:function:: entropy()

信息熵
    
返回：正态分布的信息熵, 数据类型为float32

返回类型：Variable

.. py:function:: log_prob(value)

对数概率密度函数

参数：
    - **value** (Variable) - 输入张量。
    
返回：对数概率, 数据类型与value相同

返回类型：Variable

.. py:function:: kl_divergence(other)

两个正态分布之间的KL散度。

参数：
    - **other** (Normal) - Normal的实例。
    
返回：两个正态分布之间的KL散度, 数据类型为float32

返回类型：Variable






