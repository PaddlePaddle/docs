.. _cn_api_fluid_layers_nce:

nce
-------------------------------


.. py:function:: paddle.fluid.layers.nce(input, label, num_total_classes, sample_weight=None, param_attr=None, bias_attr=None, num_neg_samples=None, name=None, sampler='uniform', custom_dist=None, seed=0, is_sparse=False)




计算并返回噪音对比估计损失值（ noise-contrastive estimation training loss）。
请参考 `Noise-contrastive estimation: A new estimation principle for unnormalized statistical models
<http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_
该层默认使用均匀分布进行抽样。

参数:
    - **input** (Variable) -  输入变量, 2-D 张量，形状为 [batch_size, dim]，数据类型为 float32 或者 float64。
    - **label** (Variable) -  标签，2-D 张量，形状为 [batch_size, num_true_class]，数据类型为 int64。
    - **num_total_classes** (int) - 所有样本中的类别的总数。
    - **sample_weight** (Variable，可选) - 存储每个样本权重，shape 为 [batch_size, 1] 存储每个样本的权重。每个样本的默认权重为1.0。
    - **param_attr** (ParamAttr，可选) ：指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) : 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **num_neg_samples** (int) - 负样例的数量，默认值是10。
    - **name** (str，可选) - 该layer的名称，具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    - **sampler** (str，可选) – 采样器，用于从负类别中进行取样。可以是 ``uniform``, ``log_uniform`` 或 ``custom_dist`` ， 默认 ``uniform`` 。
    - **custom_dist** (nd.array， 可选) – 第0维的长度为 ``num_total_classes`` 。  如果采样器类别为 ``custom_dist`` ，则使用此参数。custom_dist[i] 是第i个类别被取样的概率。默认为 None
    - **seed** (int，可选) – 采样器使用的seed。默认为0
    - **is_sparse** (bool，可选) – 标志位，指明是否使用稀疏更新, 为 ``True`` 时 :math:`weight@GRAD` 和 :math:`bias@GRAD` 的类型会变为 SelectedRows。默认为 ``False`` 。

返回： nce loss，数据类型与 **input** 相同

返回类型: Variable


**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    window_size = 5
    words = []
    for i in range(window_size):
        words.append(fluid.data(
            name='word_{0}'.format(i), shape=[-1, 1], dtype='int64'))

    dict_size = 10000
    label_word = int(window_size / 2) + 1

    embs = []
    for i in range(window_size):
        if i == label_word:
            continue

        emb = fluid.layers.embedding(input=words[i], size=[dict_size, 32],
                        param_attr='embed', is_sparse=True)
        embs.append(emb)

    embs = fluid.layers.concat(input=embs, axis=1)
    loss = fluid.layers.nce(input=embs, label=words[label_word],
            num_total_classes=dict_size, param_attr='nce.w_0',
            bias_attr='nce.b_0')

    #or use custom distribution
    dist = np.array([0.05,0.5,0.1,0.3,0.05])
    loss = fluid.layers.nce(input=embs, label=words[label_word],
            num_total_classes=5, param_attr='nce.w_1',
            bias_attr='nce.b_1',
            num_neg_samples=3,
            sampler="custom_dist",
            custom_dist=dist)




