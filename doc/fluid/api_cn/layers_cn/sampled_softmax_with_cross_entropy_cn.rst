.. _cn_api_fluid_layers_sampled_softmax_with_cross_entropy:

sampled_softmax_with_cross_entropy
----------------------------------------------

.. py:function:: paddle.fluid.layers.sampled_softmax_with_cross_entropy(logits, label, num_samples, num_true=1, remove_accidental_hits=True, use_customized_samples=False, customized_samples=None, customized_probabilities=None, seed=0)

**Sampled Softmax With Cross Entropy Operator**

对于较大的输出类，采样的交叉熵损失Softmax被广泛地用作输出层。该运算符为所有示例采样若干个样本，并计算每行采样张量的SoftMax标准化值，然后计算交叉熵损失。

由于此运算符在内部对逻辑执行SoftMax，因此它需要未分级的逻辑。此运算符不应与SoftMax运算符的输出一起使用，因为这样会产生不正确的结果。

对于T真标签（T>=1）的示例，我们假设每个真标签的概率为1/T。对于每个样本，使用对数均匀分布生成S个样本。真正的标签与这些样本连接起来，形成每个示例的T+S样本。因此，假设逻辑的形状是[N x K]，样本的形状是[N x（T+S）]。对于每个取样标签，计算出一个概率，对应于[Jean et al., 2014]( `http://arxiv.org/abs/1412.2007 <https://arxiv.org/abs/1412.2007>`_ )中的Q(y|x)。

根据采样标签对逻辑进行采样。如果remove_accidental_hits为“真”，如果sample[i, j] 意外匹配“真”标签，则相应的sampled_logits[i, j]减去1e20，使其SoftMax结果接近零。然后用logQ(y|x)减去采样的逻辑，这些采样的逻辑和重新索引的标签被用来计算具有交叉熵的SoftMax。

参数：
        - **logits** （Variable）- 非比例对数概率，是一个二维张量，形状为[N x K]。N是批大小，K是类别号。
        - **label** （Variable）- 基本事实，是一个二维张量。label是一个张量<int64>，其形状为[N x T]，其中T是每个示例的真实标签数。
        - **num_samples** （int）- 每个示例的数目num_samples应该小于类的数目。
        - **num_true** （int）- 每个训练实例的目标类别总数。
        - **remove_accidental_hits** （bool）- 指示采样时是否删除意外命中的标签。如果为真，如果一个sample[i，j]意外地碰到了真标签，那么相应的sampled_logits[i，j]将被减去1e20，使其SoftMax结果接近零。默认值为True。
        - **use_customized_samples** （bool）- 是否使用自定义样本和可能性对logits进行抽样。
        - **customized_samples** （Variable）- 用户定义的示例，它是一个具有形状[N, T + S]的二维张量。S是num_samples，T是每个示例的真标签数。
        - **customized_probabilities** （Variable）- 用户定义的样本概率，与customized_samples形状相同的二维张量。
        - **seed** （int）- 用于生成随机数的随机种子，在采样过程中使用。默认值为0。

返回：交叉熵损失，是一个二维张量，形状为[N x 1]。

返回类型：Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid

    input = fluid.layers.data(name='data', shape=[256], dtype='float32')
    label = fluid.layers.data(name='label', shape=[5], dtype='int64')
    fc = fluid.layers.fc(input=input, size=100)
    out = fluid.layers.sampled_softmax_with_cross_entropy(
              logits=fc, label=label, num_samples=25)







