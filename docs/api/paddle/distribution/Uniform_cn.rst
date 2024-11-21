.. _cn_api_paddle_distribution_Uniform:

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

参数 low 和 high 的维度必须能够支持广播。

.. note::
    关于广播(broadcasting)机制，如您想了解更多，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

参数
:::::::::

    - **low** (int|float|list|numpy.ndarray|Tensor) - 均匀分布的下边界。数据类型为 int、float、list、numpy.ndarray 或 Tensor。
    - **high** (int|float|list|numpy.ndarray|Tensor) - 均匀分布的上边界。数据类型为 int、float、list、numpy.ndarray 或 Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
:::::::::


COPY-FROM: paddle.distribution.Uniform

方法
:::::::::

sample(shape=[], seed=0)
'''''''''

生成指定维度的样本。

**参数**

    - **shape** (Sequence[int]，可选) - 1 维列表，指定生成样本的维度。数据类型为 int32。
    - **seed** (int) - 长整型数。

**返回**

Tensor，预先设计好维度的 Tensor，数据类型为 float32。

entropy()
'''''''''

信息熵

.. math::

    entropy(low, high) = \log (high - low)

**返回**

Tensor，均匀分布的信息熵，数据类型为 float32。


log_prob(value)
'''''''''

对数概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。数据类型为 float32 或 float64。

**返回**

Tensor，对数概率，数据类型与 value 相同。


probs(value)
'''''''''

概率密度函数

**参数**

    - **value** (Tensor) - 输入 Tensor。数据类型为 float32 或 float64。

**返回**

Tensor，概率，数据类型与 value 相同。
