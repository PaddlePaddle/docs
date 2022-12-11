.. _cn_api_fluid_nets_sequence_conv_pool:

sequence_conv_pool
-------------------------------


.. py:function:: paddle.fluid.nets.sequence_conv_pool(input, num_filters, filter_size, param_attr=None, act='sigmoid', pool_type='max', bias_attr=None)




**注意：该OP的输入** ``input`` **必须是2维LoDTensor, lod_level为1，如果输入是Tensor，建议使用** :ref:`cn_api_fluid_nets_simple_img_conv_pool` **代替**

该接口由序列卷积( :ref:`cn_api_fluid_layers_sequence_conv` )和池化( :ref:`cn_api_fluid_layers_sequence_pool` )组成

参数
::::::::::::

    - **input** (Variable) - sequence_conv的输入，LoDTensor, lod_level为1，支持时间长度可变的输入序列。当前输入为shape为（T，N）的矩阵，T是mini-batch中的总时间步数，N是input_hidden_size。数据类型为float32或者float64
    - **num_filters** (int)- 卷积核的数目，整数
    - **filter_size** (int)- 卷积核的大小，整数
    - **param_attr** (ParamAttr，可选) - sequence_conv层的参数属性，类型是ParamAttr或者None。默认值为None
    - **act** (str|None，可选) - sequence_conv层的激活函数类型，字符串，可以是'relu', 'softmax', 'sigmoid'等激活函数的类型。如果设置为None，则不使用激活。默认值为'sigmoid'
    - **pool_type** (str，可选) - 池化类型，字符串。可以是'max', 'average', 'sum'或者'sqrt'。默认值为'max'
    - **bias_attr** (ParamAttr|bool，可选) – sequence_conv偏置的参数属性，类型可以是bool，ParamAttr或者None。如果设置为False，则不会向输出单元添加偏置。如果将参数设置为ParamAttr的None或one属性，sequence_conv将创建ParamAttr作为bias_attr。如果未设置bias_attr的初始化器，则初始化偏差为零。默认值为None
    - **name** (str|None，可选) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name`，默认值为None

返回
::::::::::::
经过sequence_conv和sequence_pool两个操作之后的结果所表示的Tensor，数据类型与 ``input`` 相同


返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.nets.sequence_conv_pool