.. _cn_api_fluid_nets_glu:

glu
>>>>

.. py:class:: paddle.fluid.nets.glu(input, dim=-1)

The Gated Linear Units(GLU)由切分（split），sigmoid激活函数和按元素相乘组成。沿着给定维将input拆分成两个大小相同的部分，a和b，计算如下：

.. math::

    GLU(a,b) = a\bigotimes \sigma (b)

参考 : _Language Modeling with Gated Convolutional Networks: https://arxiv.org/pdf/1612.08083.pdf

参数：
    - **input** (Variable) - 输入变量，张量或者LoDTensor
    - **dim** (int) - 拆分的维度。如果 :math: `dim<0`，拆分的维为 :math: `rank(input)+dim`。默认为-1

返回：变量，大小为输入的一半

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    data = fluid.layers.data(name="words", shape=[3, 6, 9], dtype="float32")
    output = fluid.nets.glu(input=data, dim=1)  # shape of output: [3, 3, 9]

.. _cn_api_fluid_nets_sequence_conv_pool:

sequence_conv_pool
>>>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.nets.sequence_conv_pool(input, num_filters, filter_size, param_attr=None, act='sigmoid', pool_type='max')

sequence_conv_pool由序列卷积和池化组成

参数：
    - **input** (Variable) - sequence_conv的输入，支持变量时间长度输入序列。当前输入为shape为（T，N）的矩阵，T是mini-batch中的总时间步数，N是input_hidden_size
    - **num_filters** （int）- 滤波器数
    - **filter_size** （int）- 滤波器大小
    - **param_attr** （ParamAttr) - Sequence_conv层的参数。默认：None
    - **act** （str） - Sequence_conv层的激活函数类型。默认：sigmoid
    - **pool_type** （str）- 池化类型可以是max-pooling的max，average-pooling的average，sum-pooling的sum，sqrt-pooling的sqrt。默认max

返回：序列卷积（Sequence Convolution）和池化（Pooling）的结果

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    input_dim = len(word_dict)
    emb_dim = 128
    hid_dim = 512
    data = fluid.layers.data( ame="words", shape=[1], dtype="int64", lod_level=1)
    emb = fluid.layers.embedding(input=data, size=[input_dim, emb_dim], is_sparse=True)
    seq_conv = fluid.nets.sequence_conv_pool(input=emb,
                                         num_filters=hid_dim,
                                         filter_size=3,
                                         act="tanh",
                                         pool_type="sqrt")
