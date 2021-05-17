.. _cn_api_paddle_nn_BCEWithLogitsLoss:

BCEWithLogitsLoss
-------------------------------

.. py:class:: paddle.nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None, name=None)

该OP可创建一个BCEWithLogitsLoss的可调用类，计算输入 `logit` 和标签 `label` 间的 `binary cross entropy with logits loss` 损失。

该OP结合了 `sigmoid` 操作和 :ref:`api_nn_loss_BCELoss` 操作。同时，我们也可以认为该OP是 ``sigmoid_cross_entrop_with_logits`` 和一些 `reduce` 操作的组合。

在每个类别独立的分类任务中，该OP可以计算按元素的概率误差。可以将其视为预测数据点的标签，其中标签不是互斥的。例如，一篇新闻文章可以同时关于政治，科技，体育或者同时不包含这些内容。

首先，该OP可通过下式计算损失函数：

.. math::
    Out = -Labels * \log(\sigma(Logit)) - (1 - Labels) * \log(1 - \sigma(Logit))

其中 :math:`\sigma(Logit) = \frac{1}{1 + e^{-Logit}}` ， 代入上方计算公式中:

.. math::
    Out = Logit - Logit * Labels + \log(1 + e^{-Logit})

为了计算稳定性，防止当 :math:`Logit<0` 时， :math:`e^{-Logit}` 溢出，loss将采用以下公式计算:

.. math::
    Out = \max(Logit, 0) - Logit * Labels + \log(1 + e^{-\|Logit\|})

然后，当 ``weight`` or ``pos_weight`` 不为None的时候，该算子会在输出Out上乘以相应的权重。张量 ``weight`` 给Batch中的每一条数据赋予不同权重，张量 ``pos_weight`` 给每一类的正例添加相应的权重。

最后，该算子会添加 `reduce` 操作到前面的输出Out上。当 `reduction` 为 `none` 时，直接返回最原始的 `Out` 结果。当 `reduction` 为 `mean` 时，返回输出的均值 :math:`Out = MEAN(Out)` 。当 `reduction` 为 `sum` 时，返回输出的求和 :math:`Out = SUM(Out)` 。

**注意: 因为是二分类任务，所以标签值应该是0或者1。

参数
:::::::::
    - **weight** (Tensor，可选) - 手动指定每个batch二值交叉熵的权重，如果指定的话，维度必须是一个batch的数据的维度。数据类型是float32, float64。默认值是：None。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 `BCELoss` 的均值；设置为 ``'sum'`` 时，计算 `BCELoss` 的总和；设置为 ``'none'`` 时，则返回原始loss。
    - **pos_weight** (Tensor，可选) - 手动指定正类的权重，必须是与类别数相等长度的向量。数据类型是float32, float64。默认值是：None。
    - **name** (str，可选) - 操作的名称（可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

形状
:::::::::
    - **logit** (Tensor) - :math:`[N, *]` , 其中N是batch_size， `*` 是任意其他维度。输入数据 ``logit`` 一般是线性层的输出，不需要经过 ``sigmoid`` 层。数据类型是float32、float64。
    - **label** (Tensor) - :math:`[N, *]` ，标签 ``label`` 的维度、数据类型与输入 ``logit`` 相同。
    - **output** (Tensor) - 输出的Tensor。如果 :attr:`reduction` 是 ``'none'``, 则输出的维度为 :math:`[N, *]` , 与输入 ``input`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出的维度为 :math:`[1]` 。

返回
:::::::::
   返回计算BCEWithLogitsLoss的可调用对象。

代码示例
:::::::::

.. code-block:: python

    import paddle

    logit = paddle.to_tensor([5.0, 1.0, 3.0], dtype="float32")
    label = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")
    bce_logit_loss = paddle.nn.BCEWithLogitsLoss()
    output = bce_logit_loss(logit, label)
    print(output)  # [0.45618808]