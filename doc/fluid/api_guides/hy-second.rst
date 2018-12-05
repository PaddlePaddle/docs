################
hy_initializer
################

.. _cn_api_fluid_initializer_XavierInitializer:

XavierInitializer
>>>>>>>>>>>>>>>>>>>
.. py:class:: paddle.fluid.initializer.XavierInitializer(uniform=True, fan_in=None, fan_out=None, seed=0)

该类实现Xavier权重初始化方法（ Xavier weight initializer），Xavier权重初始化方法出自Xavier Glorot和Yoshua Bengio的论文 `Understanding the difficulty of training deep feedforward neural networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

该初始化函数用于保持所有层的梯度尺度几乎一致。

在均匀分布的情况下，取值范围为[-x,x]，其中：

.. math::

    x = \sqrt{\frac{6.0}{fan\_in+fan\_out}}

正态分布的情况下，均值为0，标准差为：

.. math::
    
    x = \sqrt{\frac{2.0}{fan\_in+fan\_out}}

参数：
    - **uniform** (bool) - 是否用均匀分布或者正态分布
    - **fan_in** (float) - 用于Xavier初始化的fan_in。如果为None，fan_in沿伸自变量
    - **fan_out** (float) - 用于Xavier初始化的fan_out。如果为None，fan_out沿伸自变量
    - **seed** (int) - 随机种子

.. note::

    在大多数情况下推荐将fan_in和fan_out设置为None

**代码示例**：

.. code-block:: python

    fc = fluid.layers.fc(
        input=queries, size=10,
        param_attr=fluid.initializer.Xavier(uniform=False))


.. _cn_api_fluid_initializer_MSRAInitializer:

MSRAInitializer
>>>>>>>>>>>>>>>>>

.. py:class:: paddle.fluid.initializer.MSRAInitializer(uniform=True, fan_in=None, seed=0)

实现MSRA初始化（a.k.a. Kaiming初始化）

该类实现权重初始化方法，方法来自Kaiming He，Xiangyu Zhang，Shaoqing Ren 和 Jian Sun所写的论文: `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <https://arxiv.org/abs/1502.01852>`_ 。这是一个鲁棒性特别强的初始化方法，并且适应了非线性激活函数（rectifier nonlinearities）。

 在均匀分布中，范围为[-x,x]，其中：

.. math::

    x = \sqrt{\frac{6.0}{fan\_in}}

在正态分布中，均值为0，标准差为：

.. math::

    \sqrt{\frac{2.0}{fan\_in}}

参数：
    - **uniform** (bool) - 是否用均匀分布或正态分布
    - **fan_in** (float) - MSRAInitializer的fan_in。如果为None，fan_in沿伸自变量
    - **seed** (int) - 随机种子

.. note:: 

    在大多数情况下推荐设置fan_in为None

**代码示例**：

.. code-block:: python

    fc = fluid.layers.fc(
        input=queries, size=10,
        param_attr=fluid.initializer.MSRA(uniform=False))

##################
nets
##################

.. _cn_api_fluid_nets_glu:
glu
>>>>
.. py:class:: paddle.fluid.nets.glu(input, dim=-1)
The Gated Linear Units(GLU)由切分（split），sigmoid激活函数和按元素相乘组成。沿着给定维将input拆分成两个大小相同的部分，a和b，计算如下：

.. math::

    GLU(a,b) = a\bigotimes \sigma (b)

参考论文: `Language Modeling with Gated Convolutional Networks <https://arxiv.org/pdf/1612.08083.pdf>`_

参数：
    - **input** (Variable) - 输入变量，张量或者LoDTensor
    - **dim** (int) - 拆分的维度。如果 :math: `dim<0`，拆分的维为 :math: `rank(input)+dim`。默认为-1

返回：变量 —— 变量的大小为输入的一半

返回类型：变量（Variable）

**代码示例：**

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
    - **pool_type** （str）- 池化类型。可以是max-pooling的max，average-pooling的average，sum-pooling的sum，sqrt-pooling的sqrt。默认max

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