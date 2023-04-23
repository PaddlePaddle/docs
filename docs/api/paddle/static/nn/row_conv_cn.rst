.. _cn_api_fluid_layers_row_conv:

row_conv
-------------------------------


.. py:function:: paddle.static.nn.row_conv(input, future_context_size, param_attr=None, act=None)



行卷积（Row-convolution operator）或称之为超前卷积（lookahead convolution），最早介绍于 DeepSpeech2 论文中，双向的 RNN 在深度语音模型中很有用，它通过对整个序列执行正向和反向传递来学习序列的表示。

然而，与单向 RNNs 不同的是，在线部署和低延迟设置中，双向 RNNs 具有难度。超前卷积将来自未来子序列的信息以一种高效的方式进行计算，以改进单向递归神经网络。row convolution operator 与一维序列卷积不同，计算方法如下：

给定输入序列长度为 :math:`t` 的输入序列 :math:`X` 和输入维度 :math:`D`，以及一个大小为 :math:`context * D` 的滤波器 :math:`W`，输出序列卷积为：

.. math::
    out_i = \sum_{j=i}^{i+context-1} X_{j} · W_{j-i}

公式中：
    - :math:`out_i`：第 i 行输出变量形为[1, D]。
    - :math:`context`：下文（future context）大小
    - :math:`X_j`：第 j 行输出变量，形为[1，D]
    - :math:`W_{j-i}`：第(j-i)行参数，其形状为[1,D]。

详细请参考 `设计文档 <https://github.com/PaddlePaddle/Paddle/issues/2228#issuecomment-303903645>`_ 。

论文链接：`Deep Speech 2: End-to-End Speech Recognition in English and Mandarin <https://arxiv.org/pdf/1512.02595.pdf>`_ 。

参数
::::::::::::

    - **input** (Tensor) - 支持输入为 LodTensor 和 Tensor，输入类型可以是[float32, float64]，它支持可变时间长度的输入序列。当输入 input 为 LodTensor 时，其内部 Tensor 是一个具有形状(T x N)的矩阵，其中 T 是这个 mini batch 中的总的 timestep，N 是输入数据维数。当输入 input 为 Tensor 时，其形状为(B x T x N)的三维矩阵，B 为 mini batch 大小，T 为每个 batch 输入中的最大 timestep，N 是输入数据维数。
    - **future_context_size** (int) - 下文大小。请注意，卷积核的 shape 是[future_context_size + 1, N]，N 和输入 input 的数据维度 N 保持一致。
    - **param_attr** (ParamAttr) -  参数的属性，包括名称、初始化器等。
    - **act** (str) - 非线性激活函数。

返回
::::::::::::
表示 row_conv 计算结果的 Tensor，数据类型、维度和输入 input 相同。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.row_conv
