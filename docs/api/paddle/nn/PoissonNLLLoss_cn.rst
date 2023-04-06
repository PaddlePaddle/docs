.. _cn_api_paddle_nn_PoissonNLLLoss:

PoissonNLLLoss
-------------------------------

.. py:class:: paddle.nn.PoissonNLLLoss(log_input=False, full=False, eps=1e-8, reduction='mean', name=None)

该接口可创建一个 PoissonNLLLoss 可调用类，计算输入 :attr:`input` 和标签 :attr:`label` 间的 `Poisson negative log likelihood loss` 损失。该 loss 适用于真实标签服从于泊松分布时，即

.. math::
    \text{label} \sim \mathrm{Poisson}(\text{input})

该损失函数的数学计算公式如下：
当 `log_input` 设置为 `True` 时，损失函数的数学计算公式为：

.. math::
    \text{loss}(\text{input}, \text{label}) = \text{input} - \text{label} * \log(\text{input}+\text{eps}) + \log(\text{label!})

其中 `eps` 是 ``True`` 时使用的常数小量，使得 loss 计算过程中不会导致对 0 求对数情况的出现。
当 `log_input` 设置为 `False` 时，损失函数的数学计算公式为：

.. math::
    \text{loss}(\text{input}, \text{label}) = \exp(\text{input}) - \text{label} * \text{input} + \log(\text{label!})

损失函数中的最后一项可以使用 Stirling 公式近似，该近似项的计算公式为

.. math::
    \text{label} * \log(\text{label}) - \text{label} + 0.5 * \log(2 * \pi * \text{label})

将 label 和每个元素都为 1 的同样形状的张量比较，对 label 值超过 1 的索引处考虑此项近似，对 label 的值小于等于 1 的索引处设置此项近似为 0 进行遮盖。

参数
:::::::::
    - **log_input** (bool，可选) - 输入是否为对数函数映射后结果，默认值为 ``True``。
    - **full** (bool，可选) - 是否在损失计算中包括 Stirling 近似项。默认值为 ``False``。
    - **eps** (float，可选) - 在 :attr:`log_input` 为 ``True`` 时使用的常数小量。默认值为 1e-8。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有 ``none``、``mean`` 和 ``sum``。默认为 ``mean``，计算 ``mini-batch`` loss 均值。设置为 ``sum`` 时，计算 ``mini-batch`` loss 的总和。设置为 ``none`` 时，则返回 loss Tensor。默认值下为 ``mean``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为 None。

形状
:::::::::
    - **input** (Tensor) - 输入 `Tensor`，其形状为 :math:`[N, *]` ，其中 :math:`*` 表示任何数量的额外维度。数据类型为 float32 或 float64。
    - **label** (Tensor) - 标签 :attr:`Tensor`， 形状、数据类型和 :attr:`input` 相同。

返回
:::::::::

    - **output** (Tensor) - 输入 `input` 和 `label` 间的 `Poisson negative log likelihood loss` 损失。如果 `reduction` 为 `'none'`，则输出 Loss 形状为 `[N, *]`。如果 `reduction` 为 `'sum'` 或者 `'mean'`，则输出 Loss 形状为 `'[1]'` 。

代码示例
:::::::::

COPY-FROM: paddle.nn.PoissonNLLLoss
