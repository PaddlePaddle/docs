.. _cn_api_fluid_layers_match_matrix_tensor:

match_matrix_tensor
-------------------------------

.. py:function:: paddle.fluid.layers.match_matrix_tensor(x, y, channel_num, act=None, param_attr=None, dtype='float32', name=None)

计算两个长度可变词序列的语义匹配矩阵，给一个长度为n的问题A，和一个长度为m的标题B，输入形状为[n, h]和[m, h]，h为hidden_size。如果channel_num设置为3，将会生成一个形为[h, 3, h]的参数可学习的矩阵W。接着语义匹配矩阵将会通过A * W * B.T = [n, h]*[h, 3, h]*[h, m] = [n, 3, m]来计算A和B。可学习参数矩阵W在这个过程中相当于一个全链接层。如果提供了激活函数，相关激活函数将会被用到输出中。x和y应当为LodTensor并且仅支持一个level LoD。

给一个1-level LoDTensor x:

    x.lod =  [[2,                     3,                               ]]

    x.data = [[0.3, 0.1], [0.2, 0.3], [0.5, 0.6], [0.7, 0.1], [0.3, 0.4]]

    x.dims = [5, 2]

y是一个Tensor:

    y.lod =  [[3,                                 1,       ]]

    y.data = [[0.1, 0.2], [0.3, 0.7], [0.9, 0.2], [0.4, 0.1]]

    y.dims = [4, 2]

channel_num设为2，我们就可以得到一个 1-level LoDTensor:

    out.lod =  [[12, 6]]   # where 12 = channel_num * x.lod[0][0] * y.lod[0][0]

    out.dims = [18, 1]     # where 18 = 12 + 6

参数：
    - **x** (Variable) - 1-level的输入LoDTensor。
    - **y** (Variable) - 1-level的输入LoDTensor。
    - **channel_num** (int) - 可学习参数W的通道数。
    - **act** (str,默认为None) - 激活函数。
    - **param_attr** (ParamAttr|ParamAttr的列表，默认为None) - 此层可学习参数的属性。
    - **dtype** ('float32') - w数据的数据类型。
    - **name** (str|None) - 层名，若为None，则自动设置。
    
返回：由此层指定LoD的输出

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import numpy as np
    from paddle.fluid import layers

    x_lod_tensor = layers.data(name='x', shape=[10], lod_level=1)
    y_lod_tensor = layers.data(name='y', shape=[10], lod_level=1)
    out, out_tmp = layers.match_matrix_tensor(x=x_lod_tensor, y=y_lod_tensor, channel_num=3)







