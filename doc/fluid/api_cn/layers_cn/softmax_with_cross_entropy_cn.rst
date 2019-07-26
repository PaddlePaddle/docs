.. _cn_api_fluid_layers_softmax_with_cross_entropy:

softmax_with_cross_entropy
-------------------------------

.. py:function:: paddle.fluid.layers.softmax_with_cross_entropy(logits, label, soft_label=False, ignore_index=-100, numeric_stable_mode=True, return_softmax=False, axis=-1)

使用softmax的交叉熵在输出层已被广泛使用。该函数计算输入张量在axis轴上的softmax标准化值，而后计算交叉熵。通过此种方式，可以得到更具数字稳定性的梯度值。

因为该运算是在内部进行logit上的softmax运算，所以它需要未标准化（unscaled）的logit。该运算不应该对softmax运算的输出进行操作，否则会得出错误结果。

当 ``soft_label`` 为 ``False`` 时，该运算接受互斥的硬标签，batch中的每一个样本都以为1的概率分类到一个类别中，并且仅有一个标签。

涉及到的等式如下:

1.硬标签，即 one-hot label, 每个样本仅可分到一个类别

.. math::
   loss_j =  -\text{logit}_{label_j} +\log\left(\sum_{i=0}^{K}\exp(\text{logit}_i)\right), j = 1,..., K

2.软标签，每个样本可能被分配至多个类别中

.. math::
   loss_j =  -\sum_{i=0}^{K}\text{label}_i\left(\text{logit}_i - \log\left(\sum_{i=0}^{K}\exp(\text{logit}_i)\right)\right), j = 1,...,K

3.如果 ``numeric_stable_mode`` 为真，在通过softmax和标签计算交叉熵损失前， softmax 首先经由下式计算得出：

.. math::
    max_j           &= \max_{i=0}^{K}{\text{logit}_i} \\
    log\_max\_sum_j &= \log\sum_{i=0}^{K}\exp(logit_i - max_j)\\
    softmax_j &= \exp(logit_j - max_j - {log\_max\_sum}_j)


参数:
  - **logits** (Variable) - 未标准化(unscaled)对数概率的输入张量。
  - **label** (Variable) - 真实值张量。如果 ``soft_label`` 为True，则该参数是一个和logits形状相同的的Tensor<float/double> 。如果 ``soft_label`` 为False，label是一个在axis维上形为1，其它维上与logits形对应相同的Tensor<int64>。
  - **soft_label** (bool) - 是否将输入标签当作软标签。默认为False。
  - **ignore_index** (int) - 指明要无视的目标值，使之不对输入梯度有贡献。仅在 ``soft_label`` 为False时有效，默认为kIgnoreIndex。 
  - **numeric_stable_mode** (bool) – 标志位，指明是否使用一个具有更佳数学稳定性的算法。仅在 ``soft_label`` 为 False的GPU模式下生效。若 ``soft_label`` 为 True 或者执行场所为CPU, 算法一直具有数学稳定性。 注意使用稳定算法时速度可能会变慢。默认为 True。
  - **return_softmax** (bool) – 标志位，指明是否额外返回一个softmax值， 同时返回交叉熵计算结果。默认为False。
  - **axis** (int) – 执行softmax计算的维度索引。 它应该在范围 :math:`[ -  1，rank  -  1]` 中，而 :math:`rank` 是输入logits的秩。 默认值：-1。

返回:
  - 如果 ``return_softmax`` 为 False，则返回交叉熵损失
  - 如果 ``return_softmax`` 为 True，则返回元组 (loss, softmax) ，其中softmax和输入logits形状相同；除了axis维上的形为1，其余维上交叉熵损失和输入logits形状相同

返回类型:变量或者两个变量组成的元组


**代码示例**

..  code-block:: python

  import paddle.fluid as fluid
  data = fluid.layers.data(name='data', shape=[128], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        fc = fluid.layers.fc(input=data, size=100)
        out = fluid.layers.softmax_with_cross_entropy(
        logits=fc, label=label)










