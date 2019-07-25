.. _cn_api_fluid_layers_nce:

nce
-------------------------------

.. py:function:: paddle.fluid.layers.nce(input, label, num_total_classes, sample_weight=None, param_attr=None, bias_attr=None, num_neg_samples=None, name=None, sampler='uniform', custom_dist=None, seed=0, is_sparse=False)

计算并返回噪音对比估计（ noise-contrastive estimation training loss）。
`请参考 See Noise-contrastive estimation: A new estimation principle for unnormalized statistical models
<http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_
该operator默认使用均匀分布进行抽样。

参数:
    - **input** (Variable) -  输入变量
    - **label** (Variable) -  标签
    - **num_total_classes** (int) - 所有样本中的类别的总数
    - **sample_weight** (Variable|None) - 存储每个样本权重，shape为[batch_size, 1]存储每个样本的权重。每个样本的默认权重为1.0
    - **param_attr** (ParamAttr|None) - :math:`可学习参数/nce权重` 的参数属性。如果它没有被设置为ParamAttr的一个属性，nce将创建ParamAttr为param_attr。如没有设置param_attr的初始化器，那么参数将用Xavier初始化。默认值:None
    - **bias_attr** (ParamAttr|bool|None) -  nce偏置的参数属性。如果设置为False，则不会向输出添加偏置（bias）。如果值为None或ParamAttr的一个属性，则bias_attr=ParamAtt。如果没有设置bias_attr的初始化器，偏置将被初始化为零。默认值:None
    - **num_neg_samples** (int) - 负样例的数量。默认值是10
    - **name** (str|None) - 该layer的名称(可选)。如果设置为None，该层将被自动命名
    - **sampler** (str) – 取样器，用于从负类别中进行取样。可以是 ‘uniform’, ‘log_uniform’ 或 ‘custom_dist’。 默认 ‘uniform’
    - **custom_dist** (float[]) – 一个 float[] 并且它的长度为 ``num_total_classes`` 。  如果取样器类别为‘custom_dist’，则使用此参数。 custom_dist[i] 是第i个类别被取样的概率。默认为 None
    - **seed** (int) – 取样器使用的seed。默认为0
    - **is_sparse** (bool) – 标志位，指明是否使用稀疏更新,  :math:`weight@GRAD` 和 :math:`bias@GRAD` 会变为 SelectedRows

返回： nce loss

返回类型: 变量（Variable）


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    window_size = 5
    words = []
    for i in xrange(window_size):
        words.append(fluid.layers.data(
            name='word_{0}'.format(i), shape=[1], dtype='int64'))

    dict_size = 10000
    label_word = int(window_size / 2) + 1

    embs = []
    for i in xrange(window_size):
        if i == label_word:
            continue

        emb = fluid.layers.embedding(input=words[i], size=[dict_size, 32],
                           param_attr='embed', is_sparse=True)
        embs.append(emb)

    embs = fluid.layers.concat(input=embs, axis=1)
    loss = fluid.layers.nce(input=embs, label=words[label_word],
              num_total_classes=dict_size, param_attr='nce.w_0',
              bias_attr='nce.b_0')

    # 或使用自定义分布
    dist = np.array([0.05,0.5,0.1,0.3,0.05])
    loss = fluid.layers.nce(input=embs, label=words[label_word],
              num_total_classes=5, param_attr='nce.w_1',
              bias_attr='nce.b_1',
              num_neg_samples=3,
              sampler="custom_dist",
              custom_dist=dist)




