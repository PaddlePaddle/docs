.. _cn_api_nn_loss_NLLLoss:

NLLLoss
-------------------------------

.. py:class:: paddle.nn.NLLLoss(weight=None, ignore_index=-100, reduction='mean', name=None)

该接口接受输入和目标标签，并返回 'negative log likehood loss'，用C类来训练分类问题很有效。

预计损失的输入包含每个类别的 'log' 概率，当 K 维情况下，大小必须是 (batch_size, C) 或者 (batch_size, C, d1, d2, …, dK),且 K>=1 ，损失的标签应为 [0，C-1] 范围内的类别索引，其中 C 为类别数。如果指定了 'ignore_index' ，则指定的目标值对输入梯度没有作用。

如果提供 `weight` 参数的话，它是一个 `1-D` 的 tensor，里面的值对应类别的权重。当你的训练集样本
不均衡的话，使用这个参数是非常有用的。

该损失函数的数学计算公式如下：

当 `reduction` 设置为 `none` 时，损失函数的数学计算公式为：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore_index}\},

其中 `N` 表示 `batch_size`。如果 `reduction` 的值不是 `none` (默认为 `mean`)，那么此时损失函数
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
    - **weight** (Tensor，可选) - 指定每个类别的权重。如果给定值，必须是一维张量，其大小为 [C， ] 。数据类型为 float32 或 float64。默认为 `None`。
    - **ignore_index** (int，可选) - 指定一个忽略的标签值，此标签值不参与计算。默认值为-100。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有：`none`, `mean`, `sum`。默认为 `mean`，计算 `mini-batch` loss 均值。设置为 `sum` 时，计算 `mini-batch` loss 的总和。设置为 `none` 时，则返回 loss Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

形状
:::::::::
    - **input** (Tensor) - 输入 `Tensor`，其形状为 :math:`[N, C]`，其中 `C` 为类别数。但是对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_K]`。数据类型为 float32 或 float64。
    - **label** (Tensor) - 输入 `input` 对应的标签值。其形状为 :math:`[N,]` 或者 :math:`[N, d_1, d_2, ..., d_K]`，数据类型为 int64。
    - **output** (Tensor) - 输入 `input` 和 `label` 间的 `negative log likelihood loss` 损失。如果 `reduction` 为 `'none'`，则输出 Loss 形状为 `[N, *]`。如果 `reduction` 为 `'sum'` 或者 `'mean'`，则输出 Loss 形状为 `'[1]'` 。

代码示例
:::::::::

COPY-FROM: paddle.nn.NLLLoss

输出 (input, label)
:::::::::
    定义每次调用时执行的计算。应被所有子类覆盖。

参数
:::::::::
    - **inputs** (tuple) - 未压缩的tuple参数。
    - **kwargs** (dict) - 未压缩的字典参数。