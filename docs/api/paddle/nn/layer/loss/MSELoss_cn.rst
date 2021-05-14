.. _cn_api_paddle_nn_MSELoss:

MSELoss
-------------------------------

.. py:function:: paddle.nn.MSELoss(reduction='mean')

该OP用于计算预测值和目标值的均方差误差。

对于预测值input和目标值label：

当reduction为'none'时：

.. math::
    Out = (input - label)^2

当`reduction`为`'mean'`时：

.. math::
    Out = \operatorname{mean}((input - label)^2)

当`reduction`为`'sum'`时：

.. math::
    Out = \operatorname{sum}((input - label)^2)

参数：
    - **reduction** (str, 可选) - 约简方式，可以是 'none' | 'mean' | 'sum'。设为'none'时不使用约简，设为'mean'时返回loss的均值，设为'sum'时返回loss的和。

形状:
    - **input** (Tensor) - 预测值，维度为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor。数据类型为float32或float64。
    - **label** (Tensor) - 目标值，维度为 :math:`[N_1, N_2, ..., N_k]` 的多维Tensor。数据类型为float32或float64。
    

返回：变量（Tensor）, 预测值和目标值的均方差, 数值类型与输入相同


**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle

    input_data = np.array([1.5]).astype("float32")
    label_data = np.array([1.7]).astype("float32")

    mse_loss = paddle.nn.MSELoss()
    input = paddle.to_tensor(input_data)
    label = paddle.to_tensor(label_data)
    output = mse_loss(input, label)
    print(output)
    # [0.04000002]
