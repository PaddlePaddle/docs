.. _cn_api_fluid_layers_huber_loss:

huber_loss
-------------------------------

.. py:function:: paddle.fluid.layers.huber_loss(input, label, delta)





该OP计算输入（input）与标签（label）之间的Huber损失。Huber损失是常用的回归损失之一，相较于平方误差损失，Huber损失减小了对异常点的敏感度，更具鲁棒性。

当输入与标签之差的绝对值大于delta时，计算线性误差：

.. math::
        huber\_loss = delta * (label - input) - 0.5 * delta * delta

当输入与标签之差的绝对值小于delta时，计算平方误差：

.. math::
        huber\_loss = 0.5 * (label - input) * (label - input)


参数
::::::::::::

  - **input** （Variable） - 输入的预测数据，维度为[batch_size, 1] 或[batch_size]的Tensor。数据类型为float32或float64。
  - **label** （Variable） - 输入的真实标签，维度为[batch_size, 1] 或[batch_size]的Tensor。数据类型为float32或float64。
  - **delta** （float） -  Huber损失的阈值参数，用于控制Huber损失对线性误差或平方误差的侧重。数据类型为float32。

返回
::::::::::::
 计算出的Huber损失，数据维度和数据类型与label相同的Tensor。

返回类型
::::::::::::
 Variable



代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.huber_loss