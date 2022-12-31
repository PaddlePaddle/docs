.. _cn_api_fluid_layers_sequence_conv:

sequence_conv
-------------------------------


.. py:function:: paddle.static.nn.sequence_conv(input, num_filters, filter_size=3, filter_stride=1, padding=True, padding_start=None, bias_attr=None, param_attr=None, act=None, name=None)

.. note::
1. 该 API 的输入只能是带有 LoD 信息的 Tensor，如果您需要处理的输入是普通的 Tensor 类型，请使用 :ref:`paddle.nn.functional.conv2d <cn_api_nn_functional_conv2d>` 。
2. 参数 ``padding`` 为无用参数，将在未来的版本中被移除。


在给定的卷积参数下（如卷积核数目、卷积核大小等），对输入的变长序列（sequence）Tensor 进行卷积操作。默认情况下，该 OP 会自适应地在每个输入序列的两端等长地填充全 0 数据，以确保卷积后的序列输出长度和输入长度一致。支持通过配置 ``padding_start`` 参数来指定序列填充的行为。

::

    这里详细介绍数据填充操作的细节：
    对于一个 min-batch 为 2 的变长序列输入，分别包含 3 个、1 个时间步（time_step），
    假设输入 input 是一个[4, N]的 float 类型 Tensor，为了方便，这里假设 N = 2
        input.data = [[1, 1],
                      [2, 2],
                      [3, 3],
                      [4, 4]]
        input.lod = [[0, 3, 4]]

    即输入 input 总共有 4 个词，每个词被表示为一个 2 维向量。

    Case1:

    若 padding_start = -1，filter_size = 3，
    则两端填充数据的长度分别为：
        up_pad_len = max(0, -padding_start) = 1
        down_pad_len = max(0, filter_size + padding_start - 1) = 1

    则以此填充后的输入数据为：
        data_aftet_padding = [[0, 0, 1, 1, 2, 2],
                              [1, 1, 2, 2, 3, 3],
                              [2, 2, 3, 3, 0, 0],
                              [0, 0, 4, 4, 0, 0]]

    它将和卷积核矩阵相乘得到最终的输出，假设 num_filters = 3：
        output.data = [[ 0.3234, -0.2334,  0.7433],
                       [ 0.5646,  0.9464, -0.1223],
                       [-0.1343,  0.5653,  0.4555],
                       [ 0.9954, -0.1234, -0.1234]]
        output.shape = [4, 3]     # 3 = num_filters
        output.lod = [[0, 3, 4]]  # 保持不变



参数
:::::::::

    - **input** (Variable) - 维度为 :math:`（M, K)` 的二维 Tensor，仅支持 lod_level 为 1。其中 M 是 mini-batch 的总时间步数，K 是输入的 ``hidden_size`` 特征维度。数据类型为 float32 或 float64。
    - **num_filters** (int) - 滤波器的数量。
    - **filter_size** (int，可选) - 滤波器的高度（H）；不支持指定滤波器宽度（W），宽度固定取值为输入的 ``hidden_size``。默认值为 3。
    - **filter_stride** (int，可选) - 滤波器每次移动的步长。目前只支持取值为 1，默认为 1。
    - **padding** (bool，可选) - **此参数不起任何作用，将在未来的版本中被移除。** 无论 ``padding`` 取值为 False 或者 True，默认地，该函数会自适应地在每个输入序列的两端等长地填充全 0 数据，以确保卷积后的输出序列长度和输入长度一致。默认填充是考虑到输入的序列长度可能会小于卷积核大小，这会导致无正确计算卷积输出。填充为 0 的数据在训练过程中不会被更新。默认为 True。
    - **padding_start** (int，可选) - 表示对输入序列填充时的起始位置，可以为负值。负值表示在每个序列的首端填充 ``|padding_start|`` 个时间步（time_step）的全 0 数据；正值表示对每个序列跳过前 ``padding_start`` 个时间步的数据。同时在末端填充 :math:`filter\_size + padding\_start - 1` 个时间步的全 0 数据，以保证卷积输出序列长度和输入长度一致。如果 ``padding_start`` 为 None，则在每个序列的两端填充 :math:`\frac{filter\_size}{2}` 个时间步的全 0 数据；如果 ``padding_start`` 设置为 0，则只在序列的末端填充 :math:`filter\_size - 1` 个时间步的全 0 数据。默认为 None。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **act** (str，可选) – 应用于输出上的激活函数，如 tanh、softmax、sigmoid，relu 等，支持列表请参考 :ref:`api_guide_activations`，默认值为 None。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
和输入序列等长的 Tensor，数据类型和输入一致，为 float32 或 float64。

代码示例
:::::::::

COPY-FROM: paddle.static.nn.sequence_conv
