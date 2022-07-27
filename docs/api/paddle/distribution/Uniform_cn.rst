.. _cn_api_distribution_Uniform:

Uniform
-------------------------------

.. py:class:: paddle.distribution.Uniform(low, high, name=None)




均匀分布

概率密度函数（pdf）为：

.. math::

    pdf(x; a, b) = \frac{1}{Z},  a <=x < b

    Z = b - a

上面的数学公式中：

:math:`low = a` 。
:math:`high = b` 。
:math:`Z`：正态分布常量。

参数low和high的维度必须能够支持广播。

参数
:::::::::

    - **low** (int|float|list|numpy.ndarray|Tensor) - 均匀分布的下边界。数据类型为int、float、list、numpy.ndarray或Tensor。
    - **high** (int|float|list|numpy.ndarray|Tensor) - 均匀分布的上边界。数据类型为int、float、list、numpy.ndarray或Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
:::::::::


COPY-FROM: paddle.distribution.Uniform

方法
:::::::::

sample(shape, seed=0)
'''''''''

生成指定维度的样本。

**参数**

    - **shape** (list) - 1维列表，指定生成样本的维度。数据类型为int32。
    - **seed** (int) - 长整型数。

**返回**

Tensor，预先设计好维度的张量，数据类型为float32。

entropy()
'''''''''

信息熵

.. math::

    entropy(low, high) = \log (high - low)

**返回**

Tensor，均匀分布的信息熵，数据类型为float32。


log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor) - 输入张量。数据类型为float32或float64。

**返回**

Tensor，对数概率，数据类型与value相同。


probs(value)
'''''''''

概率密度函数

**参数**

    - **value** (Tensor) - 输入张量。数据类型为float32或float64。

**返回**

Tensor，概率，数据类型与value相同。
