.. _cn_api_fluid_layers_log_loss:

log_loss
-------------------------------

.. py:function:: paddle.nn.functional.log_loss(input, label, epsilon=0.0001, name=None)




**负 log loss 层**

对输入的预测结果和目标标签进行计算，返回负对数损失值。

.. math::

    Out = -label * \log{(input + \epsilon)} - (1 - label) * \log{(1 - input + \epsilon)}


参数
::::::::::::

  - **input** (Tensor) – 形为 :math:`[N, 1]` 的二维 Tensor，其中 :math:`N` 为 batch 大小。该输入是由前驱算子计算得来的概率，数据类型是 float32。
  - **label** (Tensor) – 形为 :math:`[N, 1]` 的二维 Tensor，真值标签，其中 :math:`N` 为 batch 大小，数据类型是 float32。
  - **epsilon** (float，可选) – 一个很小的数字，以保证数值的稳定性，默认值为 1e-4。
  - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
::::::::::::

Tensor，形状为 :math:`[N, 1]`，数据类型为 float32。

代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.log_loss
