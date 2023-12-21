.. _cn_api_paddle_distribution_Gamma:

Gamma
-------------------------------

.. py:class:: paddle.distribution.Gamma(rate)

伽马分布

伽马分布的概率密度满足一下公式：

.. math::

    f(x; \alpha, \beta, x > 0) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha-1}e^{-\beta x}

    \Gamma(\alpha)=\int_{0}^{\infty} x^{\alpha-1} e^{-x} \mathrm{~d} x, (\alpha>0)

上面数学公式中：

    :math:`concentration=\alpha`：表示集中参数。

    :math:`rate=\beta`：表示率参数。


参数
::::::::::::

    - **concentration** (float|Tensor) - 率参数，该值必须大于零。支持 Broadcast 语义。当参数类型为 Tensor 时，表示批量创建多个不同参数的分布，``batch_shape`` (参考 :ref:`cn_api_paddle_distribution_Distribution` 基类) 为参数。

    - **rate** (float|Tensor) - 率参数，该值必须大于零。支持 Broadcast 语义。当参数类型为 Tensor 时，表示批量创建多个不同参数的分布，``batch_shape`` (参考 :ref:`cn_api_paddle_distribution_Distribution` 基类) 为参数。

代码示例
::::::::::::

COPY-FROM: paddle.distribution.Gamma


属性
:::::::::

mean
'''''''''
伽马分布的均值。


variance
'''''''''
伽马分布的方差。


方法
:::::::::

prob(value)
'''''''''
伽马分布的概率密度函数。

**参数**

    - **value** (float|Tensor) - 输入值。


**返回**

    - **Tensor** - value 对应的概率密度。


log_prob(value)
'''''''''
伽马分布的对数概率密度函数。

**参数**

    - **value** (float|Tensor) - 输入值。

**返回**

    - **Tensor** - value 对应的对数概率密度。


entropy()
'''''''''
伽马分布的信息熵。

**返回**

    - Tensor: 信息熵。


kl_divergence(other)
'''''''''
两个伽马分布之间的 KL 散度。

**参数**

    - **other** (Geometric) - Gamma 的实例。

**返回**

    - Tensor: 两个伽马分布之间的 KL 散度。


sample(shape)
'''''''''
随机采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int], optional) - 采样的样本维度。

**返回**

    - **Tensor** - 指定维度的样本数据。数据类型为 float32。


rsample(shape)
'''''''''
重参数化采样，生成指定维度的样本。

**参数**

    - **shape** (Sequence[int], optional) - 重参数化采样的样本维度。

**返回**

    - **Tensor** - 指定维度的样本数据。数据类型为 float32。
