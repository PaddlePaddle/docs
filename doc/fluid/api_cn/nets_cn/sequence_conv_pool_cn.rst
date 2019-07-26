.. _cn_api_fluid_nets_sequence_conv_pool:

sequence_conv_pool
-------------------------------

.. py:function:: paddle.fluid.nets.sequence_conv_pool(input, num_filters, filter_size, param_attr=None, act='sigmoid', pool_type='max', bias_attr=None)

sequence_conv_pool由序列卷积和池化组成

参数：
    - **input** (Variable) - sequence_conv的输入，支持变量时间长度输入序列。当前输入为shape为（T，N）的矩阵，T是mini-batch中的总时间步数，N是input_hidden_size
    - **num_filters** （int）- 滤波器数
    - **filter_size** （int）- 滤波器大小
    - **param_attr** （ParamAttr) - Sequence_conv层的参数。默认：None
    - **act** （str） - Sequence_conv层的激活函数类型。默认：sigmoid
    - **pool_type** （str）- 池化类型。可以是max-pooling的max，average-pooling的average，sum-pooling的sum，sqrt-pooling的sqrt。默认max
    - **bias_attr** (ParamAttr|bool|None) – sequence_conv偏置的参数属性。如果设置为False，则不会向输出单元添加偏置。如果将参数设置为ParamAttr的None或one属性，sequence_conv将创建ParamAttr作为bias_attr。如果未设置bias_attr的初始化器，则初始化偏差为零。默认值:None。

返回：序列卷积（Sequence Convolution）和池化（Pooling）的结果


返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    input_dim = 100 #len(word_dict)
    emb_dim = 128
    hid_dim = 512
    data = fluid.layers.data( name="words", shape=[1], dtype="int64", lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
    seq_conv = fluid.nets.sequence_conv_pool(input=emb,
                                         num_filters=hid_dim,
                                         filter_size=3,
                                         act="tanh",
                                         pool_type="sqrt")








