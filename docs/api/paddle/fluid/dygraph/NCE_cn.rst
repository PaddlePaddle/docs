.. _cn_api_fluid_dygraph_NCE:

NCE
-------------------------------

.. py:class:: paddle.fluid.dygraph.NCE(num_total_classes, dim, param_attr=None, bias_attr=None, num_neg_samples=None, sampler='uniform', custom_dist=None, seed=0, is_sparse=False, dtype="float32")




该接口用于构建 ``NCE`` 类的一个可调用对象，具体用法参照 ``代码示例``。其中实现了 ``NCE`` 损失函数的功能，其默认使用均匀分布进行抽样，计算并返回噪音对比估计（ noise-contrastive estimation training loss）。更多详情请参考：`Noise-contrastive estimation: A new estimation principle for unnormalized statistical models <http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>`

参数
::::::::::::

    - **num_total_classes** (int) - 所有样本中的类别的总数。
    - **dim** (int) - 输入的维度（一般为词嵌入的维度）。
    - **sample_weight** (Variable，可选) - 维度为\[batch_size, 1\]，存储每个样本的权重。每个样本的默认权重为1.0。默认值：None。
    - **param_attr** (ParamAttr，可选) - 指定权重参数属性的对象。默认值为None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选) - 指定偏置参数属性的对象。默认值为None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **num_neg_samples** (int，可选) - 负样本的数量。默认值：10。
    - **sampler** (str，可选) – 指明采样器的类型，用于从负类别中进行采样。可以是 ``uniform`` 、 ``log_uniform`` 或 ``custom_dist``。默认值：``uniform`` 。
    - **custom_dist** (float[]，可选) – float[] 类型的数据，并且它的长度为 ``num_total_classes``。如果采样器类别为 ``custom_dist``，则使用此参数。custom_dist\[i\]是第i个类别被取样的概率。默认值：None
    - **seed** (int，可选) – 采样器使用的随机种子。默认值：0。
    - **is_sparse** (bool，可选) – 指明是否使用稀疏更新，如果为True， :math:`weight@GRAD` 和 :math:`bias@GRAD` 会变为 SelectedRows。默认值：False。
    - **dtype** (str，可选) - 数据类型，可以为"float32"或"float64"。默认值："float32"。

返回
::::::::::::
无

代码示例
::::::::::::

..  code-block:: python


    import numpy as np
    import paddle.fluid as fluid

    window_size = 5
    dict_size = 20
    label_word = int(window_size // 2) + 1
    inp_word = np.array([[1], [2], [3], [4], [5]]).astype('int64')
    nid_freq_arr = np.random.dirichlet(np.ones(20) * 1000).astype('float32')

    with fluid.dygraph.guard():
        words = []
        for i in range(window_size):
            words.append(fluid.dygraph.base.to_variable(inp_word[i]))

        emb = fluid.Embedding(
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
        nce = fluid.NCE(
                    num_total_classes=dict_size,
                    dim=embs3.shape[1],
                    num_neg_samples=2,
                    sampler="custom_dist",
                    custom_dist=nid_freq_arr.tolist(),
                    seed=1,
                    param_attr='nce.w',
                    bias_attr='nce.b')

        wl = fluid.layers.unsqueeze(words[label_word], axes=[0])
        nce_loss3 = nce(embs3, wl)

属性
::::::::::::
属性
::::::::::::
weight
'''''''''

本层的可学习参数，类型为 ``Parameter``

bias
'''''''''

本层的可学习偏置，类型为 ``Parameter``
