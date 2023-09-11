.. _cn_api_paddle_nn_functional_hinge_embedding_loss:

hinge_embedding_loss
-------------------------------

.. py:class:: paddle.nn.functional.hinge_embedding_loss(input, label, margin=1.0, reduction='mean', name=None)

计算输入 input 和标签 label（包含 1 和 -1） 间的 `hinge embedding loss` 损失。

该损失通常用于度量输入 input 和标签 label 是否相似或不相似，例如可以使用 L1 成对距离作为输入 input，通常用于学习非线性嵌入或半监督学习。

对于有 :math:`n` 个样本的 mini-batch，该损失函数的数学计算公式如下：

.. math::
    l_n = \begin{cases}
        x_n, & \text{if}\; y_n = 1,\\
        \max \{0, \Delta - x_n\}, & \text{if}\; y_n = -1,
    \end{cases}

其中，:math:`x` 是 input，:math:`y` 是 label，:math:`\Delta` 是 margin。总的 loss 计算如下：

.. math::
    \ell(x, y) = \begin{cases}
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
    \end{cases}

其中，:math:`L = \{l_1,\dots,l_N\}^\top`。

参数
:::::::::
    - **input** (Tensor): - 输入的 Tensor，维度是 [N, *]，其中 N 是 batch size， `*` 是任意数量的额外维度。数据类型为：float32、float64。
    - **label** (Tensor): - 标签，维度是 [N, *]，与 ``input`` 相同，Tensor 中的值应该只包含 1 和 -1。数据类型为：float32、float64。
    - **margin** (float，可选): - 当 label 为 -1 时，该值决定了小于 `margin` 的 input 才需要纳入 `hinge embedding loss` 的计算。默认为 1.0 。
    - **reduction** (str，可选): - 指定应用于输出结果的计算方式，可选值有：``'none'``, ``'mean'``, ``'sum'``。默认为 ``'mean'``，计算 `hinge embedding loss` 的均值；设置为 ``'sum'`` 时，计算 `hinge embedding loss` 的总和；设置为 ``'none'`` 时，则返回 `hinge embedding loss`。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    Tensor，计算 HingeEmbeddingLoss 后的损失值。

形状
:::::::::
    - **input** (Tensor): - 输入的 Tensor，维度是 [N, *]，其中 N 是 batch size， `*` 是任意数量的额外维度。数据类型为：float32、float64。
    - **label** (Tensor): - 标签，维度是 [N, *]，与 ``input`` 相同，应该只包含 1 和 -1。数据类型为：float32、float64。
    - **output** (Tensor): - 输入 ``input`` 和标签 ``label`` 间的 `hinge embedding loss` 损失。如果 `reduction` 是 ``'none'``，则输出 Loss 的维度为 [N, *]，与输入 ``input`` 相同。如果 `reduction` 是 ``'mean'`` 或 ``'sum'``，则输出 Loss 的维度为 []。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.hinge_embedding_loss
