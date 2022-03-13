.. _cn_api_paddle_nn_TripletMarginLoss:

TripletMarginLoss
-------------------------------

.. py:class:: paddle.nn.TripletMarginLoss(margin: float = 1.0, p: float = 2., eps: float = 1e-6, swap: bool = False,reduction: str = 'mean')

该OP可创建一个TripletMarginLoss的可调用类，计算输入 `input` 和 `positive` 和 `negative` 间的 `triplet margin loss` 损失。


损失函数按照下列公式计算

.. math::
    L(input, pos, neg) = \max \{d(input_i, pos_i) - d(input_i, neg_i) + {\rm margin}, 0\}


其中的

.. math::
    d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p


然后， ``p`` 为距离函数的范数。 ``margin`` 为（input,positive）与（input,negative）的距离间隔， ``swap`` 的内容可以看论文Learning shallow convolutional feature descriptors with triplet losses by V. Balntas, E. Riba et al. 。

最后，该算子会添加 `reduce` 操作到前面的输出Out上。当 `reduction` 为 `none` 时，直接返回最原始的 `Out` 结果。当 `reduction` 为 `mean` 时，返回输出的均值 :math:`Out = MEAN(Out)` 。当 `reduction` 为 `sum` 时，返回输出的求和 :math:`Out = SUM(Out)` 。


参数
:::::::::
    - **p** (float，可选) - 手动指定范数，默认为2
    - **swap** (bool，可选) 
    - **margin** (float，可选) - 手动指定间距，默认为1
    - **reduction**(str,可选) -指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 `BCELoss` 的均值；设置为 ``'sum'`` 时，计算 `BCELoss` 的总和；设置为 ``'none'`` 时，则返回原始loss。
    - **name** (str，可选) - 操作的名称（可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

形状
:::::::::
    - **input** (Tensor) - :math:`[N, *]` , 其中N是batch_size， `*` 是任意其他维度。数据类型是float32、float64。
    - **positive** (Tensor) - :math:`[N, *]` ，标签 ``positive`` 的维度、数据类型与输入 ``input`` 相同。
    - **negative** (Tensor) - :math:`[N, *]` ，标签 ``negative`` 的维度、数据类型与输入 ``input`` 相同。
    - **output** (Tensor) - 输出的Tensor。如果 :attr:`reduction` 是 ``'none'``, 则输出的维度为 :math:`[N, *]` , 与输入 ``input`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出的维度为 :math:`[1]` 。

返回
:::::::::
   返回计算TripletMarginLoss的可调用对象。

代码示例
:::::::::

.. code-block:: python

      import paddle
      import paddle.nn.functional as F

      input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
      positive= paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
      negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
      loss = F.triplet_margin_loss(input, positive, negative, margin=1.0, reduction='none')

      loss = F.triplet_margin_loss(input, positive, negative, margin=1.0, reduction='mean')
