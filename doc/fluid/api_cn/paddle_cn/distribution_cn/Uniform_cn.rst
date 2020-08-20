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

参数low和high的维度必须能够支持广播。  :ref:`_user_guide_broadcasting`.

参数：
    - **low** (int|float|list|numpy.ndarray|Tensor) - 均匀分布的下边界。数据类型为float32或int。
    - **high** (int|float|list|numpy.ndarray|Tensor) - 均匀分布的上边界。数据类型为float32或int。
    - **name** (str，可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    from paddle.distribution import Normal

    paddle.disable_static()
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

    import numpy as np
    from paddle.fluid import layers
    from paddle.distribution import Uniform

    # 定义参数为float的均匀分布
    u1 = Uniform(low=3.0, high=4.0)
    # 定义参数为list的均匀分布
    u2 = Uniform(low=[1.0, 2.0],
                  high=[3.0, 4.0])
    # 通过广播的方式，定义一个均匀分布
    u3 = Uniform(low=[[1.0, 2.0],
              [3.0, 4.0]],
         high=[[1.5, 2.5],
               [3.5, 4.5]])

    # 通过广播的方式，定义一个均匀分布
    u4 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])

    # 一个完整的例子
    value_npdata = np.array([0.8], dtype="float32")
    value_tensor = layers.create_tensor(dtype="float32")
    layers.assign(value_npdata, value_tensor)

    uniform = Uniform([0.], [2.])

    sample = uniform.sample([2])
    # 一个由定义好的均匀分布随机生成的张量，维度为: [2, 1]
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





