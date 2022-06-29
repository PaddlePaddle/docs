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

:math:`loc = \mu`：平均值。
:math:`scale = \sigma`：标准差。
:math:`Z`：正态分布常量。

参数
::::::::::::

    - **loc** (float|list|numpy.ndarray|Variable) - 正态分布平均值。数据类型为float32。
    - **scale** (float|list|numpy.ndarray|Variable) - 正态分布标准差。数据类型为float32。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.Normal

参数
::::::::::::

    - **shape** (list) - 1维列表，指定生成样本的维度。数据类型为int32。
    - **seed** (int) - 长整型数。
    
返回
::::::::::::
预先设计好维度的张量，数据类型为float32

返回类型
::::::::::::
Variable

.. py:function:: entropy()

信息熵
    
返回
::::::::::::
正态分布的信息熵，数据类型为float32

返回类型
::::::::::::
Variable

.. py:function:: log_prob(value)

对数概率密度函数

参数
::::::::::::

    - **value** (Variable) - 输入张量。数据类型为float32或float64。
    
返回
::::::::::::
对数概率，数据类型与value相同

返回类型
::::::::::::
Variable

.. py:function:: kl_divergence(other)

两个正态分布之间的KL散度。

参数
::::::::::::

    - **other** (Normal) - Normal的实例。
    
返回
::::::::::::
两个正态分布之间的KL散度，数据类型为float32

返回类型
::::::::::::
Variable






