.. _cn_api_fluid_layers_npair_loss:

npair_loss
-------------------------------

.. py:function:: paddle.nn.functional.npair_loss(anchor, positive, labels, l2_reg=0.002)

参考阅读 `Improved Deep Metric Learning with Multi class N pair Loss Objective <http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf>`_

NPair 损失需要成对的数据。NPair 损失分为两部分：第一部分是对嵌入向量进行 L2 正则化；第二部分是每一对数据的相似性矩阵的每一行和映射到 ont-hot 之后的标签的交叉熵损失的和。

参数
::::::::::::

- **anchor** (Tensor) -  锚点图像的嵌入特征，形状为 [batch_size, embedding_dims] 的 2-D `Tensor`。数据类型：float32 和 float64。
- **positive** (Tensor) -  正例图像的嵌入特征，形状为 [batch_size, embedding_dims] 的 2-D `Tensor`。数据类型：float32 和 float64。
- **labels** (Tensor) - 标签向量，形状为 [batch_size] 的 1-D `Tensor`。数据类型：float32、float64 和 int64。
- **l2_reg** (float，可选) - 嵌入向量的 L2 正则化系数，默认：0.002。


返回
::::::::::::

经过 npair loss 计算之后的结果 `Tensor` 。


代码示例
::::::::::::

COPY-FROM: paddle.nn.functional.npair_loss
