.. _cn_api_paddle_nn_HingeEmbeddingLoss:

HingeEmbeddingLoss
-------------------------------

.. py:class:: paddle.nn.HingeEmbeddingLoss(delta=1.0, reduction='mean', name=None)

该接口用于创建一个HingeEmbeddingLoss的可调用类，HingeEmbeddingLoss计算输入input和标签label间的 `hinge embedding loss` 损失。

该损失通常用于度量输入input和标签label是否相似或不相似，例如可以使用 L1 成对距离作为输入input，通常用于学习非线性嵌入或半监督学习。

对于有 :math:`n` 个样本的mini-batch，该损失函数的数学计算公式如下：

.. math::
    l_n = \begin{cases}
        x_n, & \text{if}\; y_n = 1,\\
        \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
    \end{cases}

总的loss计算如下：

.. math::
    \ell(x, y) = \begin{cases}
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
    \end{cases}

其中，:math:`L = \{l_1,\dots,l_N\}^\top`.

参数
:::::::::
    - **delta** (float, 可选): - 当label为-1时，该值决定了小于 `delta` 的input才需要纳入 `hinge embedding loss` 的计算。默认为 1.0 。
    - **reduction** (str, 可选): - 指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 `HingeEmbeddingLoss` 的均值；设置为 ``'sum'`` 时，计算 `HingeEmbeddingLoss` 的总和；设置为 ``'none'`` 时，则返回 `HingeEmbeddingLoss`。
    - **name** (str，可选): - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name`。

形状
:::::::::
    - **input** (Tensor): - 输入的Tensor，维度是[N, *], 其中N是batch size， `*` 是任意数量的额外维度。数据类型为：float32、float64、int32、int64。
    - **label** (Tensor): - 标签，维度是[N, *], 与 ``input`` 相同，应该只包含 1 和 -1。数据类型为：float32、float64、int32、int64。
    - **output** (Tensor): - 输入 ``input`` 和标签 ``label`` 间的 `hinge embedding loss` 损失。如果 `reduction` 是 ``'none'``, 则输出Loss的维度为 [N, *], 与输入 ``input`` 相同。如果 `reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出Loss的维度为 [1]。

代码示例
:::::::::

.. code-block:: python

        import paddle
        import numpy as np
        import paddle.nn as nn

        input = paddle.to_tensor([[1, -2, 3], [0, -1, 2], [1, 0, 1]], dtype=paddle.float32)
        # label的元素值在 {1., -1.} 中
        label = paddle.to_tensor([[-1, 1, -1], [1, 1, 1], [1, -1, 1]], dtype=paddle.float32)

        hinge_embedding_loss = nn.HingeEmbeddingLoss(delta=1.0, reduction='none')
        loss = hinge_embedding_loss(input, label)
        print(loss)
        # Tensor([[0., -2., 0.],
        #         [0., -1., 2.],
        #         [1., 1., 1.]])

        hinge_embedding_loss = nn.HingeEmbeddingLoss(delta=1.0, reduction='mean')
        loss = hinge_embedding_loss(input, label)
        print(loss)
        # Tensor([0.22222222])
