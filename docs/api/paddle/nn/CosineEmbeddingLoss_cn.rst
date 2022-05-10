.. _cn_api_paddle_nn_CosineEmbeddingLoss:

CosineEmbeddingLoss
-------------------------------

.. py:function:: paddle.nn.CosineEmbeddingLoss(margin=0, reduction='mean')

该函数计算给定的输入input1, input2和label之间的 `CosineEmbedding` 损失，通常用于学习非线性嵌入或半监督学习

如果label=1，则该损失函数的数学计算公式如下：

    .. math::
        Out = 1 - cos(input1, input2)

如果label=-1，则该损失函数的数学计算公式如下：

    .. math::
        Out = max(0, cos(input1, input2)) - margin

参数
:::::::::
    - **margin** (float, 可选): - 可以设置的范围为[-1, 1]，建议设置的范围为[0, 0.5]。其默认为 `0` 。数据类型为int。
    - **reduction** (str, 可选): - 指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 `CosineEmbeddingLoss` 的均值；设置为 ``'sum'`` 时，计算 `CosineEmbeddingLoss` 的总和；设置为 ``'none'`` 时，则返回 `CosineEmbeddingLoss`。数据类型为string。

形状
:::::::::
    - **input1** (Tensor): - 输入的Tensor，维度是[N, M], 其中N是batch size，可为0，M是数组长度。数据类型为：float32、float64。
    - **input2** (Tensor): - 输入的Tensor，维度是[N, M], 其中N是batch size，可为0，M是数组长度。数据类型为：float32、float64。
    - **label** (Tensor): - 标签，维度是[N]，N是数组长度，数据类型为：float32、float64、int32、int64。
    - **output** (Tensor): - 输入 ``input`` 和标签 ``label`` 间的 `CosineEmbeddingLoss` 损失。如果 `reduction` 是 ``'none'``, 则输出Loss的维度为 [N], 与输入 ``input`` 相同。如果 `reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出Loss的维度为 [1]。

代码示例
:::::::::

.. code-block:: python

        import paddle

        input1 = paddle.to_tensor([[1.6, 1.2, -0.5], [3.2, 2.6, -5.8]], 'float32')
        input2 = paddle.to_tensor([[0.5, 0.5, -1.8], [2.3, -1.4, 1.1]], 'float32')
        label = paddle.to_tensor([1, -1], 'int64')

        cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='mean')
        output = cosine_embedding_loss(input1, input2, label)
        print(output) # [0.21155193]

        cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='sum')
        output = cosine_embedding_loss(input1, input2, label)
        print(output) # [0.42310387]

        cosine_embedding_loss = paddle.nn.CosineEmbeddingLoss(margin=0.5, reduction='none')
        output = cosine_embedding_loss(input1, input2, label)
        print(output) # [0.42310387, 0.        ]
