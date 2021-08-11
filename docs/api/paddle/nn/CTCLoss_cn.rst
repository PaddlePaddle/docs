.. _cn_api_paddle_nn_CTCLoss:

CTCLoss
-------------------------------

.. py:class:: paddle.nn.CTCLoss(blank=0, reduction='mean')

该接口用于计算 CTC loss。该接口的底层调用了第三方 baidu-research::warp-ctc 的实现。
也可以叫做 softmax with CTC，因为 Warp-CTC 库中插入了 softmax 激活函数来对输入的值进行归一化。

参数
:::::::::
    - **blank** (int，可选): - 空格标记的 ID 值，其取值范围为 [0，num_classes+1) 。数据类型支持int32。默认值为0。
    - **reduction** (string，可选): - 指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。设置为 ``'mean'`` 时，对 loss 值除以 label_lengths，并返回所得商的均值；设置为 ``'sum'`` 时，返回 loss 值的总和；设置为 ``'none'`` 时，则直接返回输出的 loss 值。默认值为 ``'mean'``。

形状
:::::::::
    - **logits** (Tensor): - 经过 padding 的概率序列，其 shape 必须是 [max_logit_length, batch_size, num_classes + 1]。其中 max_logit_length 是最长输入序列的长度。该输入不需要经过 softmax 操作，因为该 OP 的内部对 input 做了 softmax 操作。数据类型仅支持float32。
    - **labels** (Tensor): - 经过 padding 的标签序列，其 shape 为 [batch_size, max_label_length]，其中 max_label_length 是最长的 label 序列的长度。数据类型支持int32。
    - **input_lengths** (Tensor): - 表示输入 ``logits`` 数据中每个序列的长度，shape为 [batch_size] 。数据类型支持int64。
    - **label_lengths** (Tensor): - 表示 label 中每个序列的长度，shape为 [batch_size] 。数据类型支持int64。
    - **norm_by_times** (bool): - 当为 True 的时候，ctc grad 除以对应样本的 logits_lenth ，即 ctc_grad[B] / logits_lenth[B].数据类型支持 bool。
    - **size_average** (bool)： -  当为 True 的时候，ctc grad 除以 batch_size。数据类型支持 bool。
    - **length_average** (bool): - 当为 True 的时候，ctc grad 除以 sum(logits_lenth).数据类型支持 bool。

返回
:::::::::
``Tensor``，输入 ``logits`` 和标签 ``labels`` 间的 `ctc loss`。如果 :attr:`reduction` 是 ``'none'``，则输出 loss 的维度为 [batch_size]。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出Loss的维度为 [1]。数据类型与输入 ``logits`` 一致。

代码示例
:::::::::

.. code-block:: python

        # declarative mode
        import numpy as np
        import paddle

        # length of the longest logit sequence
        max_seq_length = 4
        #length of the longest label sequence
        max_label_length = 3
        # number of logit sequences
        batch_size = 2
        # class num
        class_num = 3

        np.random.seed(1)
        logits = np.array([[[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
                                [3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],

                                [[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
                                [5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],

                                [[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
                                [6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],

                                [[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
                                [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],

                                [[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
                                [3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]]).astype("float32")
        labels = np.array([[1, 2, 2],
                        [1, 2, 2]]).astype("int32")
        input_lengths = np.array([5, 5]).astype("int64")
        label_lengths = np.array([3, 3]).astype("int64")

        logits = paddle.to_tensor(logits)
        labels = paddle.to_tensor(labels)
        input_lengths = paddle.to_tensor(input_lengths)
        label_lengths = paddle.to_tensor(label_lengths)

        loss = paddle.nn.CTCLoss(blank=0, reduction='none')(logits, labels, 
            input_lengths, 
            label_lengths)
        print(loss)  #[3.9179852 2.9076521]

        loss = paddle.nn.CTCLoss(blank=0, reduction='mean')(logits, labels, 
            input_lengths, 
            label_lengths)
        print(loss)  #[1.1376063]

