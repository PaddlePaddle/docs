.. _cn_api_paddle_nn_RNNTLoss:

RNNTLoss
-------------------------------

.. py:class:: paddle.nn.RNNTLoss(blank=0, fastemit_lambda=0.001, reduction='mean', name=None)

计算 RNNT loss。该接口的底层调用了第三方 [warp-transducer](https://github.com/b-flo/warp-transducer.git) 的实现。
也可以叫做 softmax with RNNT，因为 warp-transducer 库中插入了 softmax 激活函数来对输入的值进行归一化。

参数
:::::::::
    - **blank** (int，可选) - 空格标记的 ID 值，其取值范围为 [0, num_classes + 1)。数据类型支持 int32。默认值为 0。
    - **fastemit_lambda** (float，默认 0.001) - FastEmit 的正则化参数(https://arxiv.org/pdf/2010.11148.pdf)。
    - **reduction** (str，可选) - 表示如何平均损失，候选是 ``'none'``|``'mean'``|``'sum'`` 。如果是 ``'mean'`` ，输出将是损失的总和并除以 batch_size;如果是 ``'sum'``，返回损失的总和;如果为 ``'none'``，则不应用 reduction。默认是 ``'mean'``。
    - **name** (str，可选) - 操作名称，默认为 None。

形状
:::::::::
    - **input** (Tensor) - 带填充的 logprobs 序列，是一个四维张量。张量形状为 [B, Tmax, Umax, D]，其中 Tmax 为输入 logit 序列的最长长度。数据类型应该是 float32 或 float64。
    - **label** (Tensor) - 带填充的基本真值序列，它必须是一个二维张量。张量形状为 [B, Umax]，其中 Umax 为标签序列的最长长度。数据类型必须为 int32。
    - **input_lengths** (Tensor) - 每个输入序列的长度，它应该有形状 [batch_size] 和 dtype int64。
    - **label_lengths** (Tensor) - 每个标签序列的长度，它应该有形状 [batch_size] 和 dtype int64。

返回
:::::::::
``Tensor``，输入 ``input`` 和标签 ``labels`` 间的 `rnnt loss`。如果 :attr:`reduction` 是 ``'none'``，则输出 loss 的维度为 [batch_size]。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``，则输出 Loss 的维度为 [1]。数据类型与输入的 ``input`` 一致。

代码示例
:::::::::

COPY-FROM: paddle.nn.RNNTLoss
