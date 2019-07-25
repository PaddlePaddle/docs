.. _cn_api_fluid_layers_sigmoid_cross_entropy_with_logits:

sigmoid_cross_entropy_with_logits
-------------------------------

.. py:function:: paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x, label, ignore_index=-100, name=None, normalize=False)

在类别不相互独立的分类任务中，该函数可以衡量按元素的可能性误差。可以这么认为，为单一数据点预测标签，它们之间不是完全互斥的关系。例如，一篇新闻文章可以同时关于政治，科技，体育或者同时不包含这些内容。

逻辑loss可通过下式计算：

.. math::
    loss = -Labels * log(sigma(X)) - (1 - Labels) * log(1 - sigma(X))

已知:

.. math::
    sigma(X) = \frac{1}{1 + exp(-X)}

代入最开始的式子，

.. math::
    loss = X - X * Labels + log(1 + exp(-X))

为了计算稳定性，防止 :math:`exp(-X)` 溢出，当 :math:`X<0` 时，loss将采用以下公式计算:

.. math::
    loss = max(X, 0) - X * Labels + log(1 + exp(-|X|))

输入 ``X`` 和 ``label`` 都可以携带LoD信息。然而输出仅采用输入 ``X`` 的LoD。



参数:
  - **x** (Variable) - (Tensor, 默认 Tensor<float>)，形为 N x D 的二维张量，N为batch大小，D为类别数目。该输入是一个由先前运算得出的logit组成的张量。logit是未标准化(unscaled)的log概率， 公式为 :math:`log(\frac{p}{1-p})`
  - **label** (Variable) -  (Tensor, 默认 Tensor<float>) 具有和X相同类型，相同形状的二维张量。该输入张量代表了每个logit的可能标签
  - **ignore_index** （int） - （int，默认kIgnoreIndex）指定被忽略的目标值，它不会影响输入梯度
  - **name** (basestring|None) - 输出的名称
  - **normalize** （bool） - 如果为true，则将输出除以除去ignore_index对应目标外的目标数

返回： (Tensor, 默认Tensor<float>), 形为 N x D 的二维张量，其值代表了按元素的逻辑loss

返回类型：Variable



**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    input = fluid.layers.data(
        name='data', shape=[10], dtype='float32')
    label = fluid.layers.data(
        name='data', shape=[10], dtype='float32')
    loss = fluid.layers.sigmoid_cross_entropy_with_logits(
        x=input,
        label=label,
        ignore_index=-1,
        normalize=True) # or False
    # loss = fluid.layers.reduce_sum(loss) # loss之和







