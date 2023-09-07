.. _cn_api_paddle_nn_functional_nll_loss:

nll_loss
-------------------------------
.. py:function:: paddle.nn.functional.nll_loss(input, label, weight=None, ignore_index=-100, reduction='mean', name=None)

返回 `negative log likelihood`。可在 :ref:`cn_api_paddle_nn_NLLLoss` 查看详情。

参数
:::::::::
    - **input** (Tensor) - 输入 `Tensor`，其形状为 :math:`[N, C]`，其中 `C` 为类别数。但是对于多维度的情形下，它的形状为 :math:`[N, C, d_1, d_2, ..., d_K]`。数据类型为 float32 或 float64。
    - **label** (Tensor) - 输入 x 对应的标签值。其形状为 :math:`[N,]` 或者 :math:`[N, d_1, d_2, ..., d_K]`，数据类型为 int64。
    - **weight** (Tensor，可选) - 手动指定每个类别的权重。其默认为 `None`。如果提供该参数的话，长度必须为 `num_classes`。数据类型为 float32 或 float64。
    - **ignore_index** (int，可选) - 指定一个忽略的标签值，此标签值不参与计算。默认值为-100。数据类型为 int64。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有：`none`, `mean`, `sum`。默认为 `mean`，计算 `mini-batch` loss 均值。设置为 `sum` 时，计算 `mini-batch` loss 的总和。设置为 `none` 时，则返回 loss Tensor。数据类型为 string。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
`Tensor`，返回存储表示 `negative log likelihood loss` 的损失值。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.nll_loss
