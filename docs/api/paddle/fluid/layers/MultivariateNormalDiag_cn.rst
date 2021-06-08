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

上面公式中:
  - :math:`inv` 表示： 对矩阵求逆
  - :math:`@` 表示：矩阵相乘
  - :math:`det` 表示：求行列式的值


参数：
    - **loc** (list|numpy.ndarray|Variable) - 形状为 :math:`[k]` 的多元高斯分布的均值列表。数据类型为float32。
    - **scale** (list|numpy.ndarray|Variable) - 形状为 :math:`[k, k]` 的多元高斯分布的对角协方差矩阵，且除对角元素外，其他元素取值均为0。数据类型为float32。

**代码示例**：

.. code-block:: python

    import numpy as np
    from paddle.fluid import layers
    from paddle.fluid.layers import MultivariateNormalDiag

    a_loc_npdata = np.array([0.3,0.5],dtype="float32")
    a_loc_tensor = layers.create_tensor(dtype="float32")
    layers.assign(a_loc_npdata, a_loc_tensor)


    a_scale_npdata = np.array([[0.4,0],[0,0.5]],dtype="float32")
    a_scale_tensor = layers.create_tensor(dtype="float32")
    layers.assign(a_scale_npdata, a_scale_tensor)

    b_loc_npdata = np.array([0.2,0.4],dtype="float32")
    b_loc_tensor = layers.create_tensor(dtype="float32")
    layers.assign(b_loc_npdata, b_loc_tensor)

    b_scale_npdata = np.array([[0.3,0],[0,0.4]],dtype="float32")
    b_scale_tensor = layers.create_tensor(dtype="float32")
    layers.assign(b_scale_npdata, b_scale_tensor)

    a = MultivariateNormalDiag(a_loc_tensor, a_scale_tensor)
    b = MultivariateNormalDiag(b_loc_tensor, b_scale_tensor)
    
    a.entropy()
    # [2.033158] with shape: [1]
    b.entropy()
    # [1.7777451] with shaoe: [1]

    a.kl_divergence(b)
    # [0.06542051] with shape: [1]


.. py:function:: kl_divergence(other)

计算相对于另一个多元高斯分布的KL散度

参数：
    - **other** (MultivariateNormalDiag) - 输入的另一个多元高斯分布。数据类型为float32。
    
返回：相对于另一个多元高斯分布的KL散度，数据类型为float32

返回类型：Variable

.. py:function:: entropy()

信息熵
    
返回：多元高斯分布的信息熵，数据类型为float32

返回类型：Variable







