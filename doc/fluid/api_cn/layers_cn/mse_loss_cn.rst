.. _cn_api_fluid_layers_mse_loss:

mse_loss
-------------------------------

.. py:function:: paddle.fluid.layers.mse_loss(input,label)



该OP用于计算预测值和目标值的均方差误差。

对于预测值input和目标值label，公式为：

.. math::

    Out = MEAN((input-label)^{2})

参数：
    - **input** (Tensor) - 预测值，维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维Tensor，其中最后一维D是类别数目。数据类型为float32或float64。
    - **label** (Tensor) - 目标值，维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维Tensor，其中最后一维D是类别数目。数据类型为float32或float64。

返回：Tensor, 预测值和目标值的均方差


**代码示例**：

.. code-block:: python

    import paddle
    input = paddle.to_tensor([1.1, 1.9])
    label = paddle.to_tensor([1.0, 2.0])
    output = paddle.fluid.layers.mse_loss(input, label)
    print(output.numpy())
    # [0.01]


