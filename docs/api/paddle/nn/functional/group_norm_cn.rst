.. _cn_api_paddle_nn_functional_group_norm:

group_norm
-------------------------------

.. py:function:: paddle.nn.functional.group_norm(x, num_groups, epsilon=1e-05, weight=None, bias=None, data_format='NCHW', name=None)

对输入 ``x`` 进行组归一化， 计算公式如下：

.. math::
    y = \frac{x - E(x)}{\sqrt(Var(x)+\epsilon)} \ast \gamma + \beta

- :math::`x`: 形状为 [批大小，通道数，\*]，其中通道数必须是 ``num_groups`` 的整数倍
- :math::`E(x)`, :math::`Var(x)`: 每一组中 ``x`` 的均值和方差
- :math::`\epsilon`: 为防止方差除零增加的一个很小的值
- :math::`\gamma`: 权重，形状为 [通道数]
- :math::`\beta`: 偏置，形状为 [通道数]

更多详情请参考：`Group Normalization <https://arxiv.org/abs/1803.08494>`_ 。

参数
::::::::::::

    - **x** (Tensor) - 输入 Tensor，形状为 [批大小，通道数，\*]。
    - **num_groups** (int) - 从通道中分离出来的 ``group`` 的数目。
    - **epsilon** (float，可选) - 为防止方差除零，增加一个很小的值。默认值：1e-05。
    - **weight** (Tensor，可选) - 权重的 Tensor，形状为 [通道数]，默认为 None。
    - **bias** (Tensor，可选) - 偏置的 Tensor，形状为 [通道数]，默认为 None。
    - **data_format** (string，可选) - 只支持 “NCHW” [num_batches，channels，height，width] 格式。默认值：“NCHW”。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
    ``Tensor``，输出形状与 ``x`` 一致。

代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.group_norm
