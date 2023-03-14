.. _cn_api_nn_functional_nll_loss:

nll_loss
-------------------------------
.. py:function:: paddle.nn.functional.gaussian_nll_loss(input, label, variance, full=False, epsilon=1e-6, reduction='mean', name=None)

返回 `gaussian negative log likelihood loss`。可在 :ref:`_cn_api_paddle_nn_GaussianNLLLoss` 查看详情。

参数
:::::::::
- **input** (Tensor)：输入 :attr:`Tensor`，其形状为 :math:`[N, *]`，其中 :math:`*` 表示任何数量的额外维度。将被拟合成为高斯分布。数据类型为 float32 或 float64。
- **label** (Tensor)：输入 :attr:`Tensor`， 形状、数据类型和 :attr:`input` 相同。为高斯分布的样本。数据类型为 float32 或 float64。
- **variance** (Tensor): 输入 :attr:`Tensor`，形状和数据类型 :attr:`input` 相同。正方差样本。数据类型为 float32 或 float64。
- **full** (bool) - 是否在损失计算中包括常数项。默认情况下为 False。
- **epsilon** (float) - 用于限制 variance 的值，使其不会导致除 0 的出现。默认值为 1e-6
- **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有 ``none``、``mean`` 和 ``sum``。默认为 ``mean``，计算 ``mini-batch`` loss 均值。设置为 `sum` 时，计算 `mini-batch` loss 的总和。设置为 ``none`` 时，则返回 loss Tensor。
- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
`Tensor`，返回存储表示 `gaussian negative log likelihood loss` 的损失值。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.nll_loss
