.. _cn_api_paddle_nn_AdaptiveLogSoftmaxWithLoss:

AdaptiveLogSoftmaxWithLoss
-------------------------------

.. py:function:: paddle.nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value=4., head_bias=False, name=None)

AdaptiveLogSoftmax是一种近似策略，提出于 `Efficient softmax approximation for GPUs <https://arxiv.org/abs/1609.04309>`_ ，适用于训练具有大输出空间的模型。当标签分布高度不平衡时，这种方法非常有效，例如NLP模型中，词频分布大致遵循Zipf定律，大多数文章是由少数的单词组成的。

AdaptiveLogSoftmax根据标签的频率将其划分为多个簇，这些簇可能包含不同数量的目标。此外，低频标签的集群会被编码为相对低维的嵌入以加快计算速度。对于每个batch，只计算至少存在一个目标的簇。

AdaptiveLogSoftmax的想法是，频繁访问的簇（如第一个簇，包含最高频的标签）也应该计算成本低廉，也就是说包含少量的标签。

`cutoffs` 应是递增排序的整数序列。它控制划分簇的数量和位置。例如，设置 `cutoffs=[10，100，1000]` 意味着前10个目标将分配给AdaptiveLogSoftmax的 `head` ，目标 `11，12，…，100` 将分配给第一个簇，目标 `101，102，…，1000` 将分配给第二个簇，而目标 `1001，1002，…，n_class-1` 将分配给最后一个簇。
`div_value` 是一个非负数，用于计算每个额外簇的大小。例如，设置 `div_value=4` 意味着簇的大小为 :math:`\lfloor\frac{in\_features}{div\_value^idx})\rfloor` ，此处idx是每个簇的索引，低频簇的索引更大，索引从1开始。
`head_bias` 如果为`True`，则对`head`层添加bias，论文作者给出的实现中，这个参数设置为`False`。

参数
:::::::::
    - **in_features** (int): 输入Tensor中特征个数
    - **n_classes** (int): 目标类别数量。
    - **cutoffs** (list): 簇的分界值，list类型，其中第i个元素表示第i+1个类别的分界值。
    - **div_value** (float，可选): 用作计算词群大小的指数，默认值为4.0。
    - **head_bias** (bool，可选): 如果为``True``，则对``head``层添加bias。默认值为``False``。

形状
:::::::::
    - **input** (Tensor): 形状为 :math:`(N, \texttt{in\_features})`,N为批大小。
    - **target** (Tensor): 形状为:math:`(N)`，其中 :math:`0 <= \texttt{target[i]} <= \texttt{n\_classes}`
    - **output1** (Tensor): 输出目标，形状为:math:`(N)`
    - **output2** (Scalar): 损失值

返回
:::::::::
计算AdaptiveLogSoftmaxWithLoss的可调用对象


代码示例
:::::::::

.. code-block:: python

        # AdaptiveLogSoftmaxWithLoss
        import paddle
        import paddle.nn as nn
        import paddle.nn.AdaptiveLogSoftmaxWithLoss

        class RNNModel(nn.Layer):
            def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
                super(RNNModel_with_adaptive, self).__init__()
                self.drop = nn.Dropout(dropout)
                self.encoder = nn.Embedding(ntoken, ninp)
                self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, time_major=True)
                self.adaptive_loss = AdaptiveLogSoftmaxWithLoss(nhid, ntoken, cutoffs=[round(ntoken/15), 3*round(ntoken/15)])
                self.init_weights()

                self.rnn_type = rnn_type
                self.nhid = nhid
                self.nlayers = nlayers

            def init_weights(self):
                initrange = 0.1
                self.encoder.weight = paddle.create_parameter(shape=self.encoder.weight.shape, dtype='float32',
                                      default_initializer=paddle.nn.initializer.Uniform(-initrange, initrange))

            def forward(self, input, hidden):
                emb = self.drop(self.encoder(input))
                output, hidden = self.rnn(emb, hidden)
                output = self.drop(output)
                return output, hidden

            def init_hidden(self, bsz):
                return (paddle.zeros([self.nlayers, bsz, self.nhid]),
                        paddle.zeros([self.nlayers, bsz, self.nhid]))

