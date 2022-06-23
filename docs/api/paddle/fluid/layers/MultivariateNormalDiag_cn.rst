.. _cn_api_fluid_layers_MultivariateNormalDiag:

MultivariateNormalDiag
-------------------------------

.. py:class:: paddle.fluid.layers.MultivariateNormalDiag(loc, scale)




多元高斯分布

概率密度函数（pdf）为：

.. math::

    pdf(x; loc, scale) = \frac{e^{-\frac{||y||^2}{2}}}{Z}
    
    y = inv(scale) @ (x - loc)
    
    Z = (2\pi )^{0.5k} |det(scale)|

上面公式中：
  - :math:`inv` 表示：对矩阵求逆
  - :math:`@` 表示：矩阵相乘
  - :math:`det` 表示：求行列式的值


参数
::::::::::::

    - **loc** (list|numpy.ndarray|Variable) - 形状为 :math:`[k]` 的多元高斯分布的均值列表。数据类型为float32。
    - **scale** (list|numpy.ndarray|Variable) - 形状为 :math:`[k, k]` 的多元高斯分布的对角协方差矩阵，且除对角元素外，其他元素取值均为0。数据类型为float32。

代码示例
::::::::::::


COPY-FROM: paddle.fluid.layers.MultivariateNormalDiag

参数
::::::::::::

    - **other** (MultivariateNormalDiag) - 输入的另一个多元高斯分布。数据类型为float32。
    
返回
::::::::::::
相对于另一个多元高斯分布的KL散度，数据类型为float32

返回类型
::::::::::::
Variable

.. py:function:: entropy()

信息熵
    
返回
::::::::::::
多元高斯分布的信息熵，数据类型为float32

返回类型
::::::::::::
Variable







