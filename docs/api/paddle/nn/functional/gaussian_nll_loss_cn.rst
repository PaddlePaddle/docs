.. _cn_api_paddle_nn_functional_gaussian_nll_loss:

gaussian_nll_loss
-------------------------------
.. py:function:: paddle.nn.functional.gaussian_nll_loss(input, label, variance, full=False, epsilon=1e-6, reduction='mean', name=None)

计算输入 :attr:`input` 、:attr:`variance` 和标签 :attr:`label` 间的 GaussianNLL 损失，
:attr:`label` 被视为高斯分布的样本，其期望 :attr:`input` 和方差 :attr:`variance` 由神经网络预测给出。
对于一个具有高斯分布的 Tensor :attr:`label`，期望 :attr:`input` 和正方差 :attr:`var` 与其损失的数学计算公式如下：

.. math::
    \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{epsilon}\right)\right) + \frac{\left(\text{input} - \text{label}\right)^2}
        {\text{max}\left(\text{var}, \ \text{epsilon}\right)}\right) + \text{const.}

参数
:::::::::
- **input** (Tensor)：输入 :attr:`Tensor`，其形状为 :math:`(N, *)` 或者 :math:`(*)`，其中 :math:`*` 表示任何数量的额外维度。将被拟合成为高斯分布。数据类型为 float32 或 float64。
- **label** (Tensor)：输入 :attr:`Tensor`，其形状为 :math:`(N, *)` 或者 :math:`(*)`，形状与 :attr:`input` 相同，或者维度与 input 相同但最后一维的大小为 1，如 :attr:`input` 的形状为： :math:`(N, 3)` 时， :attr:`input` 的形状可为 :math:`(N, 1)`，这时会进行 broadcast 操作。为服从高斯分布的样本。数据类型为 float32 或 float64。
- **variance** (Tensor): 输入 :attr:`Tensor`，其形状为 :math:`(N, *)` 或者 :math:`(*)`，形状与 :attr:`input` 相同，或者维度与 input 相同但最后一维的大小为 1，或者维度与 input 相比缺少最后一维，如 :attr:`input` 的形状为： :math:`(N, 3)` 时， :attr:`input` 的形状可为 :math:`(N, 1)` 或 :math:`(N)`，这时会进行 broadcast 操作。正方差样本，可为不同标签对应不同的方差（异方差性），也可以为同一个方差（同方差性）。数据类型为 float32 或 float64。
- **full** (bool，可选) - 是否在损失计算中包括常数项。默认情况下为 False，表示忽略最后的常数项。
- **epsilon** (float，可选) - 用于限制 variance 的值，使其不会导致除 0 的出现。默认值为 1e-6。
- **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有 ``none``、``mean`` 和 ``sum``。默认为 ``mean``，计算 ``mini-batch`` loss 均值。设置为 `sum` 时，计算 `mini-batch` loss 的总和。设置为 ``none`` 时，则返回 loss Tensor。
- **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
`Tensor`，返回存储表示 `gaussian negative log likelihood loss` 的损失值。如果 `reduction` 为 `'none'`，则输出 Loss 形状与输入相同为 `(N, *)`。如果 `reduction` 为 `'sum'` 或者 `'mean'`，则输出 Loss 形状为 `'(1)'` 。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.gaussian_nll_loss
