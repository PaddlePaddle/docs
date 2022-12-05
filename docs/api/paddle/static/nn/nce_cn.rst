.. _cn_api_fluid_layers_nce:

nce
-------------------------------


.. py:function:: paddle.static.nn.nce(input, label, num_total_classes, sample_weight=None, param_attr=None, bias_attr=None, num_neg_samples=None, name=None, sampler='uniform', custom_dist=None, seed=0, is_sparse=False)




计算并返回噪音对比估计损失值（ noise-contrastive estimation training loss）。该层默认使用均匀分布进行抽样。

论文参考：`Noise-contrastive estimation: A new estimation principle for unnormalized statistical models
<http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf>`_


参数
::::::::::::

    - **input** (Tensor) -  输入 Tensor，2-D Tensor，形状为 [batch_size, dim]，数据类型为 float32 或者 float64。
    - **label** (Tensor) -  标签，2-D Tensor，形状为 [batch_size, num_true_class]，数据类型为 int64。
    - **num_total_classes** (int) - 所有样本中的类别的总数。
    - **sample_weight** (Tensor，可选) - 存储每个样本权重，shape 为 [batch_size, 1] 存储每个样本的权重。每个样本的默认权重为 1.0。
    - **param_attr** (ParamAttr，可选)：指定权重参数属性的对象。默认值为 None，表示使用默认的权重参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **bias_attr** (ParamAttr，可选)：指定偏置参数属性的对象。默认值为 None，表示使用默认的偏置参数属性。具体用法请参见 :ref:`cn_api_fluid_ParamAttr` 。
    - **num_neg_samples** (int) - 负样例的数量，默认值是 10。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。
    - **sampler** (str，可选) – 采样器，用于从负类别中进行取样。可以是 ``uniform``, ``log_uniform`` 或 ``custom_dist``，默认 ``uniform`` 。
    - **custom_dist** (nd.array，可选) – 第 0 维的长度为 ``num_total_classes``。如果采样器类别为 ``custom_dist``，则使用此参数。custom_dist[i] 是第 i 个类别被取样的概率。默认为 None。
    - **seed** (int，可选) – 采样器使用的 seed。默认为 0。
    - **is_sparse** (bool，可选) – 标志位，指明是否使用稀疏更新，为 ``True`` 时 :math:`weight@GRAD` 和 :math:`bias@GRAD` 的类型会变为 SelectedRows。默认为 ``False`` 。

返回
::::::::::::
Tensor，nce loss，数据类型与 **input** 相同。


代码示例
::::::::::::

COPY-FROM: paddle.static.nn.nce
