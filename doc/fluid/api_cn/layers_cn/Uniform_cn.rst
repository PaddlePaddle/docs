.. _cn_api_fluid_layers_Uniform:

Uniform
-------------------------------

.. py:class:: paddle.fluid.layers.Uniform(low, high)

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
    - **low** (float|list|numpy.ndarray|Variable) - 均匀分布的下边界。
    - **high** (float|list|numpy.ndarray|Variable) - 均匀分布的上边界。

**代码示例**：

.. code-block:: python

    import numpy as np
    from paddle.fluid import layers
    from paddle.fluid.layers import Uniform

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


.. py:function:: sample(shape, seed=0)

生成指定维度的样本

参数：
    - **shape** (list) - int32的1维列表，指定生成样本的维度。
    - **seed** (int) - 长整型数。
    
返回：预先设计好维度的张量, 数据类型为float32

返回类型：Variable

.. py:function:: entropy()

信息熵
    
返回：均匀分布的信息熵, 数据类型为float32

返回类型：Variable

.. py:function:: log_prob(value)

对数概率密度函数

参数：
    - **value** (Variable) - 输入张量。
    
返回：对数概率, 数据类型与value相同

返回类型：Variable







