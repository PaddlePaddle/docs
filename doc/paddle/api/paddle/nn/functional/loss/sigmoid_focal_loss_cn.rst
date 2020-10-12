.. _cn_api_nn_functional_sigmoid_focal_loss:

sigmoid_focal_loss
-------------------------------

.. py:function:: paddle.nn.functional.sigmoid_focal_loss(logit, label, normalizer=None, alpha=0.25, gamma=2.0, reduction='sum', name=None)

`Focal Loss <https://arxiv.org/abs/1708.02002>`_ 用于解决分类任务中的前景类-背景类数量不均衡的问题。在这种损失函数，易分样本的占比被减少，而难分样本的比重被增加。例如在一阶段的目标检测任务中，前景-背景不均衡表现得非常严重。

该算子通过下式计算focal loss：

.. math::
           Out = -Labels * alpha * {(1 - \sigma(Logit))}^{gamma}\log(\sigma(Logit)) - (1 - Labels) * (1 - alpha) * {\sigma(Logit)}^{gamma}\log(1 - \sigma(Logit))

其中 :math:`\sigma(Logit) = \frac{1}{1 + \exp(-Logit)}`

当 `normalizer` 不为None时，该算子会将输出损失Out除以张量 `normalizer` ：

.. math::
           Out = \frac{Out}{normalizer}

最后，该算子会添加 `reduce` 操作到前面的输出Out上。当 `reduction` 为 ``'none'`` 时，直接返回最原始的 `Out` 结果。当 `reduction` 为 ``'mean'`` 时，返回输出的均值 :math:`Out = MEAN(Out)` 。当 `reduction` 为 ``'sum'`` 时，返回输出的求和 :math:`Out = SUM(Out)` 。

**注意**: 标签值0表示背景类（即负样本），1表示前景类（即正样本）。

参数
:::::::::
    - **logit** (Tensor) - 维度为 :math:`[N, *]` , 其中N是batch_size， `*` 是任意其他维度。输入数据 ``logit`` 一般是卷积层的输出，不需要经过 ``sigmoid`` 层。数据类型是float32、float64。
    - **label** (Tensor) - 维度为 :math:`[N, *]` ，标签 ``label`` 的维度、数据类型与输入 ``logit`` 相同。
    - **normalizer** (Tensor，可选) - 维度为 :math:`[1]` ，focal loss的归一化系数，数据类型与输入 ``logit`` 相同。若设置为None，则不会将focal loss做归一化操作（即不会将focal loss除以normalizer）。在目标检测任务中，设置为正样本的数量。默认值为None。
    - **alpha** (int|float，可选) - 用于平衡易分样本和难分样本的超参数，取值范围 :math:`[0，1]` 。默认值设置为0.25。
    - **gamma** (int|float，可选) - 用于平衡正样本和负样本的超参数，默认值设置为2.0。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有: ``'none'``, ``'mean'``, ``'sum'`` 。默认为 ``'mean'``，计算 `focal loss` 的均值；设置为 ``'sum'`` 时，计算 `focal loss` 的总和；设置为 ``'none'`` 时，则返回原始loss。
    - **name** (str，可选) - 操作的名称（可选，默认值为None）。更多信息请参见 :ref:`api_guide_Name` 。

返回
:::::::::
    - Tensor，输出的Tensor。如果 :attr:`reduction` 是 ``'none'``, 则输出的维度为 :math:`[N, *]` , 与输入 ``logit`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``, 则输出的维度为 :math:`[1]` 。

代码示例
:::::::::

.. code-block:: python

    import paddle

    logit = paddle.to_tensor([[0.97, 0.91, 0.03], [0.55, 0.43, 0.71]], dtype='float32')
    label = paddle.to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32')
    one = paddle.to_tensor([1.], dtype='float32')
    fg_label = paddle.greater_equal(label, one)
    fg_num = paddle.reduce_sum(paddle.cast(fg_label, dtype='float32'))
    output = paddle.nn.functional.sigmoid_focal_loss(logit, label, normalizer=fg_num)
    print(output.numpy())  # [0.65782464]
