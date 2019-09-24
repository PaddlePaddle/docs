.. _cn_api_fluid_layers_fsp_matrix:

fsp_matrix
-------------------------------

.. py:function:: paddle.fluid.layers.fsp_matrix(x, y)

**FSP matrix 算子**

fsp_matrix运算用于计算两个4-D Tensor特征图的求解过程（FSP）矩阵。假设特征图x的形状为 :math:`[x\_channel，h，w]` ，特征图y的形状为 :math:`[y\_channel，h，w]` ，fsp_matrix运算分两步得到x和y的fsp矩阵：

1.将x reshape到形状为 :math:`[x\_channel，h*w]` 的矩阵，将y reshape到形状为 :math:`[h*w，y\_channel]` 的矩阵。

2.对x和y做矩阵乘法得到形状为 :math:`[x\_channel，y\_channel]` 的fsp矩阵。

输出是一个batch的fsp矩阵。

参数：
    - **x** (Variable): 一个形状为 :math:`[batch\_size, x\_channel, height, width]` 的特征图Tensor, 数据类型为float32或float64。
    - **y** (Variable)：一个形状为 :math:`[batch\_size, y\_channel, height, width]` 的特征图Tensor, 数据类型为float32或float64。y_channel可以与输入（x）的x_channel不同，而其他维度必须与输入（x）相同。

返回：一个形状为 :math:`[batch\_size, x\_channel, y\_channel]` 的fsp矩阵, 是一个 3-D Tensor，数据类型与输入数据类型一致。其中，x_channel是输入x的通道数，y_channel是输入y的通道数。

返回类型：Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[3, 32, 32])
    feature_map_0 = fluid.layers.conv2d(data, num_filters=2,
                                        filter_size=3)
    feature_map_1 = fluid.layers.conv2d(feature_map_0, num_filters=2,
                                        filter_size=1)
    loss = fluid.layers.fsp_matrix(feature_map_0, feature_map_1)

