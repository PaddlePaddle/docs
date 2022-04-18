.. _cn_api_paddle_nn_MultiLabelSoftMarginLoss:

MultiLabelSoftMarginLoss
-------------------------------

.. py:class:: paddle.nn.MultiLabelSoftMarginLoss(weight:Optional=None, reduction: str = 'mean')

该OP可创建一个MultiLabelSoftMarginLoss的可调用类，计算输入 `input` 和 `label` 间的 `margin-based loss` 损失。


损失函数按照下列公式计算

.. math::
    \text{loss}(x, y) = \sum_{ij}\frac{\max(0, 1 - (x[y[j]] - x[i]))}{\text{x.size}(0)}

如果添加权重则再乘以对应的权重值


最后，该算子会添加 `reduce` 操作到前面的输出Out上。当 `reduction` 为 `none` 时，直接返回最原始的 `Out` 结果。当 `reduction` 为 `mean` 时，返回输出的均值 :math:`Out = MEAN(Out)` 。当 `reduction` 为 `sum` 时，返回输出的求和 :math:`Out = SUM(Out)` 。


参数
:::::::::
    - **weight** (Tensor，可选) - 手动设定权重，默认为None
    - **reduction**(str,可选) -指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 Loss 的均值；设置为 ``'sum'`` 时，计算 Loss 的总和；设置为 ``'none'`` 时，则返回原始Loss。
    - **name** (str，可选) - 操作的名称（可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

形状
:::::::::
    - **input** (Tensor) - :math:`[N, *]` , 其中N是batch_size， `*` 是任意其他维度。数据类型是float32、float64。
    - **label** (Tensor) - :math:`[N, *]` ，标签 ``label`` 的维度、数据类型与输入 ``input`` 相同。
    - **output** (Tensor) - 输出的Tensor。如果 :attr:`reduction` 是 ``'none'``, 则输出的维度为 :math:`[N, *]` , 与输入 ``input`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出的维度为 :math:`[1]` 。

返回
:::::::::
   返回计算MultiLabelSoftMarginLoss的可调用对象。

代码示例
:::::::::

.. code-block:: python

      import paddle
      import paddle.nn.layer as L

      input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
      label= paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
      weight = paddle.to_tensor([[0.3, 0.1, 0.16], [0.2, 1, 0.21], [0.4, 0.32, 0.03]], dtype=paddle.float32)
      multi_label_soft_margin_loss = L.MultiLabelSoftMarginLoss(weight=weight, reduction='mean')

      loss = multi_label_soft_margin_loss(input, label)
