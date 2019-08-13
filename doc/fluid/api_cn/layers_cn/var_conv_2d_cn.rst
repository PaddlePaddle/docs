.. _cn_api_fluid_layers_var_conv_2d:

var_conv_2d
-------------------------------

.. py:function:: paddle.fluid.layers.var_conv_2d(input, row, col, input_channel, output_channel, filter_size, stride=1, param_attr=None, act=None, dtype='float32', name=None)

var_conv_2d层依据给定的参数来计算输出， ``input`` 、 ``row`` 和 ``col`` 都是1-level的 ``LodTensor`` 卷积操作与普通的conv2d卷积层一样，值得注意的是，输入数据的第二个维度即input.dim[1]应该为1。
如果 ``input_channel`` 是2，并且给了如下的row lodTensor 和 col lodTensor:

.. code-block:: text

    row.lod = [[5, 4]]
    col.lod = [[6, 7]]
    输入是一个lodTensor:
    input.lod = [[60, 56]]  # where 60 = input_channel * 5 * 6
    input.dims = [116, 1]   # where 116 = 60 + 56
    如果设置 output_channel 为3, filter_size 为 [3, 3], stride 为 [1, 1]:
    output.lod = [[90, 84]] # where 90 = output_channel * [(5-1)/stride + 1] * [(6-1)/stride + 1]
    output.dims = [174, 1]  # where 174 = 90 + 84

参数:
    - **input** (Variable) – dims[1]等于1的1-level的LodTensor。
    - **row** (Variable) – 1-level的LodTensor提供height。
    - **col** (Variable) – 1-level的LodTensor提供width。
    - **input_channel** (int) – 输入通道的数目。
    - **output_channel** (int) – 输出通道的数目。
    - **filter_size** (int|tuple|None) – 过滤器尺寸。 如果是元组，则应当为两个整型数字(filter_size_H, filter_size_W)。否则，过滤器会变为正方形。
    - **stride** (int|tuple) – 步长。 如果是元组，则应当为两个整型数字(stride_H, stride_W)。否则，stride_H = stride_W = stride。默认: stride = 1.
    - **param_attr** (ParamAttr|None) – 为var_conv2d可学习的权重分配参数属性如果设置为None，或者ParamAttr的一个属性, var_conv2d将会创建ParamAttr做为param_attr。如果param_attr的初始化没有设定，参数将会以 \(Normal(0.0, std)\),进行初始化，\(std\) 为 \((\frac{2.0 }{filter\_elem\_num})^{0.5}\). 默认: None。
    - **act** (str) – 激活类型，如果设置为None，则不会激活。默认:None
    - **dtype** ('float32') – 输出与参数的数据类型
    - **name** (str|None) – 层名。如果没有设置，将会被自动命名。默认: None。


返回: 由该层指定LoD的输出变量

返回类型: 变量(Variable)

**代码示例**：

.. code-block:: python

    import numpy as np
    from paddle.fluid import layers

    x_lod_tensor = layers.data(name='x', shape=[1], lod_level=1)
    row_lod_tensor = layers.data(name='row', shape=[6], lod_level=1)
    col_lod_tensor = layers.data(name='col', shape=[6], lod_level=1)
    out = layers.var_conv_2d(input=x_lod_tensor,
                             row=row_lod_tensor,
                             col=col_lod_tensor,
                             input_channel=3,
                             output_channel=5,
                             filter_size=[3, 3],
                             stride=1)







