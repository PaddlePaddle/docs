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
:math:`Z`: 正态分布常量。

参数low和high的维度必须能够支持广播。

参数：
    - **low** (int|float|list|numpy.ndarray|Tensor) - 均匀分布的下边界。数据类型为int、float、list、numpy.ndarray或Tensor。
    - **high** (int|float|list|numpy.ndarray|Tensor) - 均匀分布的上边界。数据类型为int、float、list、numpy.ndarray或Tensor。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.distribution import Uniform

    # Without broadcasting, a single uniform distribution [3, 4]:
    u1 = Uniform(low=3.0, high=4.0)
    # 2 distributions [1, 3], [2, 4]
    u2 = Uniform(low=[1.0, 2.0], high=[3.0, 4.0])
    # 4 distributions
    u3 = Uniform(low=[[1.0, 2.0], [3.0, 4.0]],
            high=[[1.5, 2.5], [3.5, 4.5]])

    # With broadcasting:
    u4 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])

    # Complete example
    value_npdata = np.array([0.8], dtype="float32")
    value_tensor = paddle.to_tensor(value_npdata)

    uniform = Uniform([0.], [2.])

    sample = uniform.sample([2])
    # a random tensor created by uniform distribution with shape: [2, 1]
    entropy = uniform.entropy()
    # [0.6931472] with shape: [1]
    lp = uniform.log_prob(value_tensor)
    # [-0.6931472] with shape: [1]
    p = uniform.probs(value_tensor)
    # [0.5] with shape: [1]


.. py:function:: sample(shape, seed=0)

生成指定维度的样本

参数：
    - **shape** (list) - 1维列表，指定生成样本的维度。数据类型为int32。
    - **seed** (int) - 长整型数。
    
返回：预先设计好维度的张量, 数据类型为float32

返回类型：Tensor

.. py:function:: entropy()

信息熵

.. math::

    entropy(low, high) = \log (high - low)

返回：均匀分布的信息熵, 数据类型为float32

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





