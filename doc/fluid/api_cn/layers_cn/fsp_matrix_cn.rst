.. _cn_api_fluid_layers_fsp_matrix:

fsp_matrix
-------------------------------

.. py:function:: paddle.fluid.layers.fsp_matrix(x, y)

**FSP matrix op**

此运算用于计算两个特征映射的求解过程（FSP）矩阵。给定形状为[x_channel，h，w]的特征映射x和形状为[y_channel，h，w]的特征映射y，我们可以分两步得到x和y的fsp矩阵：

1.用形状[X_channel，H*W]将X重塑为矩阵，并用形状[H*W，y_channel]将Y重塑和转置为矩阵。

2.乘以x和y得到形状为[x_channel，y_channel]的fsp矩阵。

输出是一批fsp矩阵。

参数：
    - **x** (Variable): 一个形状为[batch_size, x_channel, height, width]的特征映射
    - **y** (Variable)：具有形状[batch_size, y_channel, height, width]的特征映射。Y轴通道可以与输入（X）的X轴通道不同，而其他尺寸必须与输入（X）相同。

返回：形状为[batch_size, x_channel, y_channel]的fsp op的输出。x_channel 是x的通道，y_channel是y的通道。

返回类型：fsp matrix (Variable)

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    data = fluid.layers.data(name='data', shape=[3, 32, 32])
    feature_map_0 = fluid.layers.conv2d(data, num_filters=2,
                                        filter_size=3)
    feature_map_1 = fluid.layers.conv2d(feature_map_0, num_filters=2,
                                        filter_size=1)
    loss = fluid.layers.fsp_matrix(feature_map_0, feature_map_1)






