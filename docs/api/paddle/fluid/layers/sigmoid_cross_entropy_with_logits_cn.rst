.. _cn_api_fluid_layers_sigmoid_cross_entropy_with_logits:

sigmoid_cross_entropy_with_logits
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x, label, ignore_index=-100, name=None, normalize=False)




在每个类别独立的分类任务中，该 OP 可以计算按元素的概率误差。可以将其视为预测数据点的标签，其中标签不是互斥的。例如，一篇新闻文章可以同时关于政治，科技，体育或者同时不包含这些内容。

logistic loss 可通过下式计算：

.. math::
    loss = -Labels * log(sigma(X)) - (1 - Labels) * log(1 - sigma(X))

已知：

.. math::
    sigma(X) = \frac{1}{1 + exp(-X)}

代入上方计算 logistic loss 公式中：

.. math::
    loss = X - X * Labels + log(1 + exp(-X))

为了计算稳定性，防止 :math:`exp(-X)` 溢出，当 :math:`X<0` 时，loss 将采用以下公式计算：

.. math::
    loss = max(X, 0) - X * Labels + log(1 + exp(-|X|))

输入 ``X`` 和 ``label`` 都可以携带 LoD 信息。然而输出仅采用输入 ``X`` 的 LoD。



参数
::::::::::::

  - **x** (Variable) - (Tensor，默认 Tensor<float>)，形为 N x D 的二维 Tensor，N 为 batch 大小，D 为类别数目。该输入是一个由先前运算得出的 logit 组成的 Tensor。logit 是未标准化(unscaled)的 log 概率，公式为 :math:`log(\frac{p}{1-p})`，数据类型为 float32 或 float64。
  - **label** (Variable) -  (Tensor，默认 Tensor<float>) 具有和 ``X`` 相同数据类型，相同形状的二维 Tensor。该输入 Tensor 代表了每个 logit 的可能标签。
  - **ignore_index** （int） - （int，默认 kIgnoreIndex）指定被忽略的目标值，它不会影响输入梯度。
  - **name**  (str，可选) – 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
  - **normalize** （bool） - 如果为 true，则将输出除以除去 ignore_index 对应目标外的目标数，默认为 False。

返回
::::::::::::
 Variable(Tensor，默认 Tensor<float>)，形为 N x D 的二维 Tensor，其值代表了按元素的 logistic loss，数据类型为 float32 或 float64。

返回类型
::::::::::::
变量(Variable)



代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.sigmoid_cross_entropy_with_logits
