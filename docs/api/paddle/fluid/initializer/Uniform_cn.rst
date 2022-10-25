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
:math:`Z`：正态分布常量。

参数low和high的维度必须能够支持广播。

参数
::::::::::::

    - **low** (float|list|numpy.ndarray|Variable) - 均匀分布的下边界。数据类型为float32。
    - **high** (float|list|numpy.ndarray|Variable) - 均匀分布的上边界。数据类型为float32。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.Uniform

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
均匀分布的信息熵，数据类型为float32

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







