.. _cn_api_paddle_nn_functional_adaptive_log_softmax_with_loss:

adaptive_log_softmax_with_loss
-------------------------------

.. py:function:: paddle.nn.functional.adaptive_log_softmax_with_loss(input, label, head_weight, tail_weights, cutoffs, head_bias=None, name=None)

计算自适应 logsoftmax 结果以及 input 和 label 之间的负对数似然。参数 `head_weight`、`tail_weights`、`cutoffs`和 `head_bias` 是 `AdaptiveLogSoftmaxWithLoss` 的内部成员。
请参考：:ref:`cn_api_paddle_nn_AdaptiveLogSoftmaxWithLoss`


参数
:::::::::
    - **input** (Tensor): 输入张量，数据类型为 float32 或 float64。
    - **label** (Tensor): 标签张量，数据类型为 float32 或 float64。
    - **head_weight** (Tensor): 用于线性计算的权重矩阵，数据类型为 float32 或 float64。
    - **tail_weights** (Tensor): 用于线性计算的权重矩阵，数据类型为 float32 或 float64。
    - **cutoffs** (Sequence): 用于将 label 分配到不同存储桶的截断值。
    - **head_bias** (Tensor, 可选): 用于线性计算的偏置矩阵，数据类型为 float32 或 float64。
    - **name** (str, 可选): 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
    - **output** (Tensor): - 自适应 logsoftmax 计算结果，形状为[N]。
    - **loss** (Tensor): - input 和 label 之间的 logsoftmax 损失值。

代码示例
:::::::::
COPY-FROM: paddle.nn.functional.adaptive_log_softmax_with_loss
