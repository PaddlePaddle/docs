.. _cn_api_fluid_layers_sequence_conv:

sequence_conv
-------------------------------

.. py:function:: paddle.fluid.layers.sequence_conv(input, num_filters, filter_size=3, filter_stride=1, padding=True, padding_start=None, bias_attr=None, param_attr=None, act=None, name=None)

该函数接收变长序列以及其他的卷积配置参数，把过滤器及步长应用到卷积操作上。它将会默认使用零来填充序列的边，以保证输出与输入具有相同的长度。可以通过配置参数 ``padding_start`` 来定制padding行为。

注意：
参数 ``padding`` 没有影响，未来将会失效。

我们在此举例说明padding操作的细节：

.. code-block:: python

    一个包含两个变长句子的mini-batch, 包括 3, 和 1 time-steps:
    假设输入 (X) 是一个shape为 [4, M, N] 的浮点型 LoDTensor, 并且 X->lod()[0] = [0, 3, 4].
    除此之外, 为了简化, 我们假设 M=1 and N=2.
    X = [[a1, a2;
          b1, b2;
          c1, c2]
         [d1, d2]]

    就是说输入 (X) 有4个单词，并且每个单词的维度代表是2.

    * 例1:

        如果 padding_start 为 -1 并且 filter_size 为 3.
        padding数据的长度将会按照下面来计算:
        up_pad_len = max(0, -padding_start) = 1
        down_pad_len = max(0, filter_size + padding_start - 1) = 1

        输入序列在padding之后的输出为:
        data_aftet_padding = [[0,  0,  a1, a2, b1, b2;
                               a1, a2, b1, b2, c1, c2;
                               b1, b2, c1, c2, 0,  0 ]
                              [0,  0,  d1, d2, 0,  0 ]]

        它将与过滤器的权重相乘得到最终的结果。

参数：
    - **input** (Variable) - (LoD张量）输入X是LoD张量，支持可变的时间量的长度输入序列。该LoDTensor的标记张量是一个维度为（T,N)的矩阵，其中T是mini-batch的总时间步数，N是input_hidden_size
    - **num_filters** (int) - 滤波器的数量
    - **filter_size** (int) - 滤波器的H，W为默认的隐藏尺寸。
    - **filter_stride** (int) - 滤波器的步长，当前仅支持为1。
    - **padding** (bool) - 此参数无影响，未来会移除。当前无论设置为True还是Flase都将会填充输入来确保输入与输出具有相同的长度，因为输入序列的长度可能小于过滤器尺寸，将会造成卷积结果计算错误，训练中这些padding数据不会被训练。
    - **padding_start** （int|None）- 用来指定填充序列开始的index，可以为负，负数表示在每一个实例的开始会填充|padding_start| time-steps 的全0数据。正数表示跳过每一个实例的padding_start time-steps,并且将会填充（filter_size + padding_strat - 1）time-steps的全0数据。在序列的末端保证输出与输入具有相同的长度。如果设置为None,与数据相同的长度 (filter_size/2) 将被填充到序列的两边，如果设置为0，数据的长度 (filter_size-1) 将会被填充到每个输入序列的尾端。
    - **bias_attr** (ParamAttr|bool|None) - sequence_conv偏离率参数属性。若设为False,输出单元则不加入偏离率。若设为None或ParamAttr的一个属性，sequence_conv将创建一个ParamAttr作为bias_attr。如果未设置bias_attr的初始化函数，则将bias初始化为0.默认:None
    - **param_attr** (ParamAttr|None) - 可学习参数/sequence_conv的权重参数属性。若设置为None或ParamAttr的一个属性，sequence_conv将创建ParamAttr作为param_attr。
    若未设置param_attr的初始化函数，则用Xavier初始化参数。默认:None

返回：sequence_conv的输出

返回类型：变量（Variable）

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    x = fluid.layers.data(name='x', shape=[10,10], append_batch_size=False, dtype='float32')
    x_conved = fluid.layers.sequence_conv(input=x, num_filters=2, filter_size=3, padding_start=-1)







