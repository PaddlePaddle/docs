.. _cn_api_fluid_dygraph_NCE:

NCE
-------------------------------

.. py:class:: paddle.fluid.dygraph.NCE(name_scope, num_total_classes, param_attr=None, bias_attr=None, num_neg_samples=None, sampler='uniform', custom_dist=None, seed=0, is_sparse=False)

计算并返回噪音对比估计（ noise-contrastive estimation training loss）。 
`请参考Noise-contrastive estimation: A new estimation principle for unnormalized statistical models
<http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_

该operator默认使用均匀分布进行抽样。

参数:
    - **name_scope** (str) – 该类的名称
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


    import numpy as np
    import paddle.fluid as fluid

    window_size = 5
    dict_size = 20
    label_word = int(window_size // 2) + 1
    inp_word = np.array([[[1]], [[2]], [[3]], [[4]], [[5]]]).astype('int64')
    nid_freq_arr = np.random.dirichlet(np.ones(20) * 1000).astype('float32')

    with fluid.dygraph.guard():
        words = []
        for i in range(window_size):
            words.append(fluid.dygraph.base.to_variable(inp_word[i]))

        emb = fluid.Embedding(
            'embedding',
            size=[dict_size, 32],
            param_attr='emb.w',
            is_sparse=False)

        embs3 = []
        for i in range(window_size):
            if i == label_word:
                continue

            emb_rlt = emb(words[i])
            embs3.append(emb_rlt)

        embs3 = fluid.layers.concat(input=embs3, axis=1)
        nce = fluid.NCE('nce',
                     num_total_classes=dict_size,
                     num_neg_samples=2,
                     sampler="custom_dist",
                     custom_dist=nid_freq_arr.tolist(),
                     seed=1,
                     param_attr='nce.w',
                     bias_attr='nce.b')

        nce_loss3 = nce(embs3, words[label_word])




