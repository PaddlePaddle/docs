.. _cn_api_fluid_layers_sequence_conv:

sequence_conv
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_conv(input, num_filters, filter_size=3, filter_stride=1, padding=None, bias_attr=None, param_attr=None, act=None, name=None)

该函数的输入参数中给出了滤波器和步长，通过利用输入以及滤波器和步长的常规配置来为sequence_conv创建操作符。

参数：
    - **input** (Variable) - (LoD张量）输入X是LoD张量，支持可变的时间量的长度输入序列。该LoDTensor的标记张量是一个维度为（T,N)的矩阵，其中T是mini-batch的总时间步数，N是input_hidden_size
    - **num_filters** (int) - 滤波器的数量
    - **filter_size** (int) - 滤波器大小（H和W)
    - **filter_stride** (int) - 滤波器的步长
    - **padding** (bool) - 若为真，添加填充
    - **bias_attr** (ParamAttr|bool|None) - sequence_conv偏离率参数属性。若设为False,输出单元则不加入偏离率。若设为None或ParamAttr的一个属性，sequence_conv将创建一个ParamAttr作为bias_attr。如果未设置bias_attr的初始化函数，则将bias初始化为0.默认:None
    - **param_attr** (ParamAttr|None) - 可学习参数/sequence_conv的权重参数属性。若设置为None或ParamAttr的一个属性，sequence_conv将创建ParamAttr作为param_attr。
    若未设置param_attr的初始化函数，则用Xavier初始化参数。默认:None

返回：sequence_conv的输出

返回类型：变量（Variable）

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[10,10], append_batch_size=False, dtype='float32')
    x_conved = fluid.layers.sequence_conv(x,2)







