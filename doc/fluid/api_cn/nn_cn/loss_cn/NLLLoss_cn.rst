.. _cn_api_nn_loss_NLLLoss:

NLLLoss
-------------------------------

.. py:class:: paddle.nn.loss.NLLLoss(weight=None, ignore_index=-100, reduction='mean', name=None)

该接口可创建一个NLLLoss可调用类，计算输入x和标签label间的 `negative log likelihood loss` 损失 ，可用于训练一个 `n` 类分类器。

如果提供 `weight` 参数的话，它是一个 `1-D` 的tensor, 里面的值对应类别的权重。当你的训练集样本
不均衡的话，使用这个参数是非常有用的。

该损失函数的数学计算公式如下：

当 `reduction` 设置为 `none` 时，损失函数的数学计算公式为：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore_index}\},

其中 `N` 表示 `batch_size` 。如果 `reduction` 的值不是 `none` (默认为 `mean`)，那么此时损失函数
的数学计算公式为：

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{'sum'.}
        \end{cases}

参数
:::::::::
    - **weight** (Tensor, 可选): - 手动指定每个类别的权重。其默认为 `None` 。如果提供该参数的话，长度必须为 `num_classes` 。数据类型为float32或float64。
    - **ignore_index** (int64, 可选): - 指定一个忽略的标签值，此标签值不参与计算。默认值为-100。数据类型为int64。
    - **reduction** (str, 可选): - 指定应用于输出结果的计算方式，可选值有: `none`, `mean`, `sum` 。默认为 `mean` ，计算 `mini-batch` loss均值。设置为 `sum` 时，计算 `mini-batch` loss的总和。设置为 `none` 时，则返回loss Tensor。数据类型为string。
    - **name** (str, 可选） - 操作的名称(可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

形状
:::::::::
    - **input** (Tensor): - 输入 `Tensor`, 其形状为 :math:`[N, C]` , 其中 `C` 为类别数。但是对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_K]` 。数据类型为float32或float64。
    - **label** (Tensor): - 输入 `input` 对应的标签值。其形状为 :math:`[N,]` 或者 :math:`[N, d_1, d_2, ..., d_K]`, 数据类型为int64。
    - **output** (Tensor): - 输入 `input` 和 `label` 间的 `negative log likelihood loss` 损失。如果 `reduction` 为 `'none'` ，则输出Loss形状为 `[N, *]` 。 如果 `reduction` 为 `'sum'` 或者 `'mean'` ，则输出Loss形状为 `'[1]'` 。

代码示例
:::::::::

.. code-block:: python

        import paddle
        import numpy as np

        nll_loss = paddle.nn.layer.NLLLoss()
        log_softmax = paddle.nn.LogSoftmax(axis=1)

        input_np = np.array([[0.88103855, 0.9908683 , 0.6226845 ],
                         [0.53331435, 0.07999352, 0.8549948 ],
                         [0.25879037, 0.39530203, 0.698465  ],
                         [0.73427284, 0.63575995, 0.18827209],
                         [0.05689114, 0.0862954 , 0.6325046 ]]).astype(np.float32)
        label_np = np.array([0, 2, 1, 1, 0]).astype(np.int64)

        place = paddle.CPUPlace()
        paddle.disable_static(place)
        input = paddle.to_variable(input_np)
        log_out = log_softmax(input)
        label = paddle.to_variable(label_np)
        result = nll_loss(log_out, label)
        print(result.numpy()) # [1.0720209]
