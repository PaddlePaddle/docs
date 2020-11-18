mse_loss
-------------------------------

.. py:function:: paddle.nn.functional.mse_loss(input, label, reduction='mean', name=None)

该OP用于计算预测值和目标值的均方差误差。

对于预测值input和目标值label，公式为：

当 `reduction` 设置为 ``'none'`` 时，
    
    .. math::
        Out = (input - label)^2

当 `reduction` 设置为 ``'mean'`` 时，

    .. math::
       Out = \operatorname{mean}((input - label)^2)

当 `reduction` 设置为 ``'sum'`` 时，
    
    .. math::
       Out = \operatorname{sum}((input - label)^2)


参数：
:::::::::
    - **input** (Tensor) - 预测值，维度为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor。数据类型为float32或float64。
    - **label** (Tensor) - 目标值，维度为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor。数据类型为float32或float64。

返回
:::::::::
``Tensor``, 输入 ``input`` 和标签 ``label`` 间的 `mse loss` 损失。

**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle
    mse_loss = paddle.nn.loss.MSELoss()
    input = paddle.to_tensor(1.5)
    label = paddle.to_tensor(1.7)
    output = mse_loss(input, label)
    print(output.numpy())
    # [0.04000002]

