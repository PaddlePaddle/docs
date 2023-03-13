.. _cn_api_fluid_layers_fsp_matrix:

fsp_matrix
-------------------------------

.. py:function:: paddle.fluid.layers.fsp_matrix(x, y)




**FSP matrix op**

fsp_matrix op 用于计算两个 4-D Tensor 特征图的求解过程（FSP）矩阵。假设特征图 x 的形状为 :math:`[x\_channel，h，w]`，特征图 y 的形状为 :math:`[y\_channel，h，w]` ，fsp_matrix op 分两步得到 x 和 y 的 fsp 矩阵：

1. 将 x reshape 到形状为 :math:`[x\_channel，h*w]` 的矩阵，将 y reshape 到形状为 :math:`[h*w，y\_channel]` 的矩阵。

2. 对 x 和 y 做矩阵乘法得到形状为 :math:`[x\_channel，y\_channel]` 的 fsp 矩阵。

输出是一个 batch 的 fsp 矩阵。

参数
::::::::::::

    - **x** (Variable)：一个形状为 :math:`[batch\_size, x\_channel, height, width]` 的 4-D 特征图 Tensor，数据类型为 float32 或 float64。
    - **y** (Variable)：一个形状为 :math:`[batch\_size, y\_channel, height, width]` 的 4-D 特征图 Tensor，数据类型为 float32 或 float64。y_channel 可以与输入（x）的 x_channel 不同，而其他维度必须与输入（x）相同。

返回
::::::::::::
一个形状为 :math:`[batch\_size, x\_channel, y\_channel]` 的 fsp 矩阵，是一个 3-D Tensor，数据类型与输入数据类型一致。其中，x_channel 是输入 x 的通道数，y_channel 是输入 y 的通道数。数据类型为 float32 或 float64。

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.fsp_matrix
