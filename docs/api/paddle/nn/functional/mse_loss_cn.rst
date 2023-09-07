.. _cn_api_paddle_nn_functional_mse_loss:

mse_loss
-------------------------------

.. py:function:: paddle.nn.functional.mse_loss(input, label, reduction='mean', name=None)

用于计算预测值和目标值的均方差误差。

对于预测值 input 和目标值 label，公式为：

当 `reduction` 设置为 ``'none'`` 时，

    .. math::
        Out = (input - label)^2

当 `reduction` 设置为 ``'mean'`` 时，

    .. math::
       Out = \operatorname{mean}((input - label)^2)

当 `reduction` 设置为 ``'sum'`` 时，

    .. math::
       Out = \operatorname{sum}((input - label)^2)


参数
:::::::::
    - **input** (Tensor) - 预测值，维度为 :math:`[N_1, N_2, ..., N_k]` 的多维 Tensor。数据类型为 float32 或 float64。
    - **label** (Tensor) - 目标值，维度为 :math:`[N_1, N_2, ..., N_k]` 的多维 Tensor。数据类型为 float32 或 float64。
    - **reduction** (str, 可选) - 输出的归约方法可以是'none'、'mean'或'sum'。

        - 如果 :attr:`reduction` 是 ``'mean'``，则返回减少的平均损失。
        - 如果 :attr:`reduction` 是 ``'sum'``，则返回减少的总损失。
        - 如果 :attr: `reduction` 为 `'none'`，返回未减少的损失。默认为 `'mean'`。

    - **name** (str, 可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
``Tensor``，输入 ``input`` 和标签 ``label`` 间的 `mse loss` 损失。

代码示例
:::::::::

COPY-FROM: paddle.nn.functional.mse_loss
