.. _cn_api_paddle_distribution_StudentT:

StudentT
-------------------------------

.. py:class:: paddle.distribution.StudentT(df, loc, scale, name=None)


正态分布

数学公式：

.. math::

    pdf(x; \nu, \mu, \sigma) = \frac{\Gamma[(\nu+1)/2]}{\sigma\sqrt{\nu\pi}\Gamma(\nu/2)[1+(\frac{x-\mu}{\sigma})^2/\nu]^{(1+\nu)/2}}


上面的数学公式中：

- :math:`df = \nu`：自由度；
- :math:`loc = \mu`：平移变换参数；
- :math:`scale = \sigma`：缩放变换参数；
- :math:`\Gamma(\cdot)`：gamma 函数；

参数
::::::::::::

    - **df** (float|Tensor) - t 分布的自由度，需大于 0。若输入类型是 float， :attr:`df` 会被转换成数据类型为 paddle 全局默认数据类型的 1-D tensor。若输入类型是 tensor，则支持的数据类型有 float32 或 float64。
    - **loc** (float|Tensor) - t 分布的平移变换参数。若输入类型是 float， :attr:`loc` 会被转换成数据类型为 paddle 全局默认数据类型的 1-D tensor。若输入类型是 tensor，则支持的数据类型有 float32 或 float64。
    - **scale** (float|Tensor) - t 分布的缩放变换参数，需大于 0。若输入类型是 float， :attr:`scale` 会被转换成数据类型为 paddle 全局默认数据类型的 1-D tensor。若输入类型是 tensor，则支持的数据类型有 float32 或 float64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
::::::::::::


COPY-FROM: paddle.distribution.StudentT


属性
:::::::::

mean
'''''''''

t 分布的均值

**返回**

Tensor，均值

variance
'''''''''

t 分布的方差

**返回**

Tensor，方差

方法
:::::::::

prob(value)
'''''''''

计算 value 的概率。

**参数**

    - **value** (Tensor) - 待计算值。

**返回**

Tensor，value 的概率。数据类型与 :attr:`df` 相同。


log_prob(value)
'''''''''

计算 value 的对数概率。

**参数**

    - **value** (Tensor) - 待计算值。

**返回**

Tensor，value 的对数概率。数据类型与 :attr:`df` 相同。


sample()
'''''''''

从 t 分布中生成满足特定形状的样本数据。最终生成样本形状为 ``shape+batch_shape`` 。

**参数**

    - **shape** (Sequence[int]，可选)：采样次数。

**返回**

Tensor：样本数据。其维度为 :math:`\text{sample shape} + \text{batch shape}` 。

entropy()
'''''''''

计算 t 分布的信息熵。

.. math::

    H = \log(\frac{\Gamma(\nu/2)\Gamma(1/2) \sigma \sqrt{\nu}}{\Gamma[(1+\nu)/2]}) + \frac{(1+\nu)}{2} \cdot \{\psi[(1+\nu)/2] - \psi(\nu/2)\}

上面的数学公式中：

- :math:`\nu`：自由度；
- :math:`\Gamma(\cdot)`：gamma 函数；
- :math:`\psi(\cdot)`：digamma 函数；

**返回**

t 分布的信息熵，数据类型与 :attr:`df` 相同。
