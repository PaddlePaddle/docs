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

参数low和high应该为可支持广播的shape。

参数：
    - **low** (float|list|numpy.ndarray|Variable) - 均匀分布的较低边界。
    - **high** (float|list|numpy.ndarray|Variable) - 均匀分布的较高边界。

**代码示例**：

.. code-block:: python

    from paddle.fluid import layers
    from paddle.fluid.layers import Uniform

    # 一个未广播的单独的均匀分布 [3, 4]:
    u1 = Uniform(low=3.0, high=4.0)
    # 两个分布 [1, 3], [2, 4]
    u2 = Uniform(low=[1.0, 2.0],
                  high=[3.0, 4.0])
    # 4个分布
    u3 = Uniform(low=[[1.0, 2.0],
              [3.0, 4.0]],
         high=[[1.5, 2.5],
               [3.5, 4.5]])

    # 广播:
    u4 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])

    # 作为输入的变量
    dims = 3

    low = layers.data(name='low', shape=[dims], dtype='float32')
    high = layers.data(name='high', shape=[dims], dtype='float32')
    values = layers.data(name='values', shape=[dims], dtype='float32')

    uniform = Uniform(low, high)

    sample = uniform.sample([2, 3])
    entropy = uniform.entropy()
    lp = uniform.log_prob(values)


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






