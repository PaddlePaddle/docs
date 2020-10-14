.. _cn_api_fluid_layers_mse_loss:

mse_loss
-------------------------------

.. py:function:: paddle.fluid.layers.mse_loss(input,label)




该OP用于计算预测值和目标值的均方差误差。

对于预测值input和目标值label，公式为：

.. math::

    Out = MEAN((input-label)^{2})

参数：
    - **input** (Variable) - 预测值，维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维Tensor，其中最后一维D是类别数目。数据类型为float32或float64。
    - **label** (Variable) - 目标值，维度为 :math:`[N_1, N_2, ..., N_k, D]` 的多维Tensor，其中最后一维D是类别数目。数据类型为float32或float64。

返回：预测值和目标值的均方差

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    y = fluid.data(name='y', shape=[1], dtype='float32')
    y_predict = fluid.data(name='y_predict', shape=[1], dtype='float32')
    cost = fluid.layers.mse_loss(input=y_predict, label=y)



