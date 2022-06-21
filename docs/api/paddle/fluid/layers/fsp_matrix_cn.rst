.. _cn_api_fluid_layers_fsp_matrix:

fsp_matrix
-------------------------------

.. py:function:: paddle.fluid.layers.fsp_matrix(x, y)




**FSP matrix op**

fsp_matrix op用于计算两个4-D Tensor特征图的求解过程（FSP）矩阵。假设特征图x的形状为 :math:`[x\_channel，h，w]`，特征图y的形状为 :math:`[y\_channel，h，w]` ，fsp_matrix op分两步得到x和y的fsp矩阵：

1. 将x reshape到形状为 :math:`[x\_channel，h*w]` 的矩阵，将y reshape到形状为 :math:`[h*w，y\_channel]` 的矩阵。

2. 对x和y做矩阵乘法得到形状为 :math:`[x\_channel，y\_channel]` 的fsp矩阵。

输出是一个batch的fsp矩阵。

参数
::::::::::::

    - **x** (Variable)：一个形状为 :math:`[batch\_size, x\_channel, height, width]` 的 4-D 特征图Tensor，数据类型为float32或float64。
    - **y** (Variable)：一个形状为 :math:`[batch\_size, y\_channel, height, width]` 的 4-D 特征图Tensor，数据类型为float32或float64。y_channel可以与输入（x）的x_channel不同，而其他维度必须与输入（x）相同。

返回
::::::::::::
一个形状为 :math:`[batch\_size, x\_channel, y\_channel]` 的fsp矩阵，是一个 3-D Tensor，数据类型与输入数据类型一致。其中，x_channel是输入x的通道数，y_channel是输入y的通道数。数据类型为float32或float64。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.fsp_matrix