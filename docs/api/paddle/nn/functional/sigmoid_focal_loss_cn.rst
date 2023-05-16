.. _cn_api_nn_functional_sigmoid_focal_loss:

sigmoid_focal_loss
-------------------------------

.. py:function:: paddle.nn.functional.sigmoid_focal_loss(logit, label, normalizer=None, alpha=0.25, gamma=2.0, reduction='sum', name=None)

`Focal Loss <https://arxiv.org/abs/1708.02002>`_ 用于解决分类任务中的前景类-背景类数量不均衡的问题。在这种损失函数，易分样本的占比被减少，而难分样本的比重被增加。例如在一阶段的目标检测任务中，前景-背景不均衡表现得非常严重。

该算子通过下式计算 focal loss：

.. math::
           Out = -Labels * alpha * {(1 - \sigma(Logit))}^{gamma}\log(\sigma(Logit)) - (1 - Labels) * (1 - alpha) * {\sigma(Logit)}^{gamma}\log(1 - \sigma(Logit))

其中 :math:`\sigma(Logit) = \frac{1}{1 + \exp(-Logit)}`

当 `normalizer` 不为 None 时，该算子会将输出损失 Out 除以 Tensor `normalizer` ：

.. math::
           Out = \frac{Out}{normalizer}

最后，该算子会添加 `reduce` 操作到前面的输出 Out 上。当 `reduction` 为 ``'none'`` 时，直接返回最原始的 `Out` 结果。当 `reduction` 为 ``'mean'`` 时，返回输出的均值 :math:`Out = MEAN(Out)`。当 `reduction` 为 ``'sum'`` 时，返回输出的求和 :math:`Out = SUM(Out)` 。

**注意**：标签值 0 表示背景类（即负样本），1 表示前景类（即正样本）。

参数
:::::::::
    - **logit** (Tensor) - 维度为 :math:`[N, *]`，其中 N 是 batch_size， `*` 是任意其他维度。输入数据 ``logit`` 一般是卷积层的输出，不需要经过 ``sigmoid`` 层。数据类型是 float32、float64。
    - **label** (Tensor) - 维度为 :math:`[N, *]`，标签 ``label`` 的维度、数据类型与输入 ``logit`` 相同，取值范围 :math:`[0，1]`。
    - **normalizer** (Tensor，可选) - 维度为 :math:`[]` ，focal loss 的归一化系数，数据类型与输入 ``logit`` 相同。若设置为 None，则不会将 focal loss 做归一化操作（即不会将 focal loss 除以 normalizer）。在目标检测任务中，设置为正样本的数量。默认值为 None。
    - **alpha** (int|float，可选) - 用于平衡正样本和负样本的超参数，取值范围 :math:`[0，1]`。默认值设置为 0.25。
    - **gamma** (int|float，可选) - 用于平衡易分样本和难分样本的超参数，默认值设置为 2.0。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有：``'none'``, ``'mean'``, ``'sum'``。默认为 ``'mean'``，计算 `focal loss` 的均值；设置为 ``'sum'`` 时，计算 `focal loss` 的总和；设置为 ``'none'`` 时，则返回原始 loss。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - Tensor，输出的 Tensor。如果 :attr:`reduction` 是 ``'none'``，则输出的维度为 :math:`[N, *]`，与输入 ``logit`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``，则输出的维度为 :math:`[]` 。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.sigmoid_focal_loss
