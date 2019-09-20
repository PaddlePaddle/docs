.. _cn_api_fluid_layers_sequence_conv:

sequence_conv
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_conv(input, num_filters, filter_size=3, filter_stride=1, padding=True, padding_start=None, bias_attr=None, param_attr=None, act=None, name=None)

**注意：该OP的输入只能是LoDTensor，如果您需要处理的输入是Tensor类型，请使用conv2d函数（fluid.layers.** :ref:`cn_api_fluid_layers_conv2d` **）或者conv3d函数（fluid.layers.** :ref:`cn_api_fluid_layers_conv3d` **）。**

该OP在给定的卷积参数下，如卷积核大小、卷积步长等，对输入的变长序列（sequence）LoDTensor进行卷积操作。默认情况下，该OP会自适应地在每个输入序列的两端等长地填充全0数据，以确保卷积后的序列输出长度和输入长度一致。支持通过配置 ``padding_start`` 参数来指定序列填充的行为。

**提示：** 参数 ``padding`` 为无用参数，将在未来的版本中被移除。

::

    这里将详细介绍数据填充操作的细节：
    对于一个min-batch为2的变长序列输入，分别包含3个、1个时间步（time_step），
    假设输入input是一个[4, M, N]的float类型LoDTensor，input.lod()[0] = [0, 3, 4]
    为了方便，这里假设M=1，N=2
        input.data = [[a1, a2;
                      b1, b2;
                      c1, c2]
                     [d1, d2]]
    即输入input总共有4个词，每个词被表示为一个2维向量。

    Case1:

    若 padding_start = -1，filter_size = 3，
    则两端填充数据的长度分别为：
        up_pad_len = max(0, -padding_start) = 1
        down_pad_len = max(0, filter_size + padding_start - 1) = 1

    则以此填充后的输入数据为：
        data_aftet_padding = [[0,  0,  a1, a2, b1, b2;
                               a1, a2, b1, b2, c1, c2;
                               b1, b2, c1, c2, 0,  0 ]
                              [0,  0,  d1, d2, 0,  0 ]]
    它将和卷积核矩阵相乘得到最终的输出。


参数：
    - **input** (LoDTensor) - 维度为 :math:`（T,N)` 的LoDTensor，仅支持lod_level为1。其中T是mini-batch的总时间步数，N是输入的 ``hidden_size`` 特征维度。
    - **num_filters** (int) - 滤波器的数量。
    - **filter_size** (int) - 滤波器的高度（H），滤波器宽度（W）固定为输入的 ``hidden_size`` 。
    - **filter_stride** (int) - 滤波器每次移动的步长。目前只支持取值为1，默认为1。
    - **padding** (bool) - **此参数不起任何作用，将在未来的版本中被移除。** 无论 ``padding`` 取值为 ``False`` 或者 ``True`` ，默认地，该函数会自适应地在每个输入序列的两端等长地填充全0数据，以确保卷积后的输出序列长度和输入长度一致。默认填充是考虑到输入的序列长度可能会小于卷积核大小，这会导致无正确计算卷积输出。填充为0的数据在训练过程中不会被更新。
    - **padding_start** (int) - 表示对输入序列填充时的起始位置，可以为负值。负值表示在每个序列的首端填充 ``|padding_start|`` 个时间步（time_step）的全0数据；正值表示对每个序列跳过前 ``padding_start`` 个时间步的数据。同时在末端填充 :math:`filter\_size + padding\_start - 1` 个时间步的全0数据，以保证卷积输出序列长度和输入长度一致。如果 ``padding_start`` 为 ``None`` ，则在每个序列的两端填充 :math:`\\frac{filter\_size}{2}` 个时间步的全0数据；如果 ``padding_start`` 设置为0，则只在序列的末端填充 :math:`filter\_size - 1` 个时间步的全0数据。默认为 ``None`` 。
    - **bias_attr** (ParamAttr) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **param_attr** (ParamAttr) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。

返回：和输入序列等长的LoDTensor，数据类型和输入一致。

返回类型：Variable

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[10,10], append_batch_size=False, dtype='float32')
    x_conved = fluid.layers.sequence_conv(x,2)







