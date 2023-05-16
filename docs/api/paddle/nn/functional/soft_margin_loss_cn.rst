.. _cn_api_paddle_nn_functional_soft_margin_losss:

soft_margin_loss
-------------------------------

.. py:function:: paddle.nn.functional.soft_margin_loss(input, label, reduction='mean', name=None)

计算输入 `input` 和 `label` 间的二分类损失。


损失函数按照下列公式计算

.. math::
    \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}


最后，添加 `reduce` 操作到前面的输出 Out 上。当 `reduction` 为 `none` 时，直接返回最原始的 `Out` 结果。当 `reduction` 为 `mean` 时，返回输出的均值 :math:`Out = MEAN(Out)` 。当 `reduction` 为 `sum` 时，返回输出的求和 :math:`Out = SUM(Out)` 。


参数
:::::::::
    - **input** (Tensor) - :math:`[N, *]` ，其中 N 是 batch_size， `*` 是任意其他维度。数据类型是 float32、float64。
    - **label** (Tensor) - :math:`[N, *]` ，标签 ``label`` 的维度、数据类型与输入 ``input`` 相同。
    - **reduction** (str，可选) - 指定应用于输出结果的计算方式，可选值有： ``'none'``、 ``'mean'``、 ``'sum'`` 。默认为 ``'mean'``，计算 Loss 的均值；设置为 ``'sum'`` 时，计算 Loss 的总和；设置为 ``'none'`` 时，则返回原始 Loss。
    - **name** (str，可选) - 操作的名称（可选，默认值为 None）。更多信息请参见 :ref:`api_guide_Name` 。


返回
:::::::::
    - 输出的结果 Tensor。如果 :attr:`reduction` 是 ``'none'``, 则输出的维度为 :math:`[N, *]` ，与输入 ``input`` 的形状相同。如果 :attr:`reduction` 是 ``'mean'`` 或 ``'sum'``，则输出的维度为 :math:`[]` 。


代码示例
:::::::::
COPY-FROM: paddle.nn.functional.soft_margin_loss
