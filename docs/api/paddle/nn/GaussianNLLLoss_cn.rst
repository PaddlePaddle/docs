.. _cn_api_paddle_nn_GaussianNLLLoss:

SmoothL1Loss
-------------------------------

.. py:class:: paddle.nn.GaussianNLLLoss(full=False, eps=1e-6, reduction='mean', name=None)

计算输入 :attr:`input` 和标签 :attr:`target`, :attr:`var` 间的 GaussianNLL 损失，
:attr:`target` 被视为高斯分布的样本，其期望和方差由神经网络预测给出。对于一个 :attr:`target` 张量建模为具有高斯分布的张量的期望值 :attr:`input` 和张量的正方差 :attr:`var`，数学计算公式如下：

.. math::
    \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{target}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

参数
::::::::::

    - **full** (bool) - 是否在损失计算中包括常数项。默认情况下为False。
    - **delta** (float) - 用于限制var的值，使其不会导致除0的出现。默认值为1e-6
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有 ``none``、``mean`` 和 ``sum``。默认为 ``mean``，计算 ``mini-batch`` loss 均值。设置为 `sum` 时，计算 `mini-batch` loss 的总和。设置为 ``none`` 时，则返回 loss Tensor。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

输入
::::::::::

    - **input** (Tensor)：输入 :attr:`Tensor`，其形状为 :math:`[N, *]`，其中 :math:`*` 表示任何数量的额外维度。
    - **target** (Tensor)：输入 :attr:`Tensor`， 形状、数据类型和 :attr:`input` 相同。
    - **var** (Tensor): 输入 :attr:`Tensor`，形状和 :attr:`input` 相同。

返回
:::::::::

Tensor，计算 `GaussianNLLLoss` 后的损失值。若 :attr:`reduction` 为 ``none``，则与输入形状相同。


代码示例
:::::::::

COPY-FROM: paddle.nn.GaussianNLLLoss
