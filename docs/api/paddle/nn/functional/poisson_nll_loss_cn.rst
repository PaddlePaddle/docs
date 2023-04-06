.. _cn_api_paddle_nn_functional_poisson_nll_loss:

poisson_nll_loss
-------------------------------
.. py:function:: paddle.nn.functional.poisson_nll_loss(input, label, log_input=False, full=False, eps=1e-8, reduction='mean', name=None)

返回 `poisson negative log likelihood`。可在 :ref:`cn_api_paddle_nn_PoissonNLLLoss` 查看详情。

参数
:::::::::
    - **input** (Tensor) - 输入 :attr:`Tensor`，对应泊松分布的期望，其形状为 :math:`[N, *]`，其中 :math:`*` 表示任何数量的额外维度。数据类型为 float32 或 float64。
    - **label** (Tensor) - 标签 :attr:`Tensor`， 形状、数据类型和 :attr:`input` 相同。
    - **log_input** (bool，可选) - 输入是否为对数函数映射后结果，如果为 ``True``，则 loss 当中第一项的计算公式为

    .. math::
        \text{input} - \text{label} * \log(\text{input}+\text{eps})

    其中 :attr:`eps` 为数值稳定使用的常数小量。
    如果为 ``False``，则 loss 的计算公式为

    .. math::
        \exp(\text{input}) - \text{label} * \text{input}

    默认值为 ``True``。

    - **full** (bool，可选) - 是否在损失计算中包括 Stirling 近似项。该近似项的计算公式为

    .. math::
        \text{label} * \log(\text{label}) - \text{label} + 0.5 * \log(2 * \pi * \text{label})

    默认值为 ``False``。

    - **eps** (float，可选) - 在 :attr:`log_input` 为 ``True`` 时使用的常数小量，使得 loss 计算过程中不会导致对 0 求对数情况的出现。默认值为 1e-8。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有 ``none``、``mean`` 和 ``sum``。默认为 ``mean``，计算 ``mini-batch`` loss 均值。设置为 ``sum`` 时，计算 ``mini-batch`` loss 的总和。设置为 ``none`` 时，则返回 loss Tensor。默认值下为 ``mean``。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
`Tensor`，返回存储表示 `poisson negative log likelihood loss` 的损失值。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.poisson_nll_loss
