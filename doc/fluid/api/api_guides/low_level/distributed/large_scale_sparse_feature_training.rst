.. _api_guide_large_scale_sparse_feature_training:

###################
大规模稀疏特征模型训练
###################


模型配置和训练
=============

embedding被广泛应用在各种网络结构中，尤其是文本处理相关的模型。在某些场景，例如推荐系统或者搜索引擎中，
embedding的feature id可能会非常多，当feature id达到一定数量时，embedding参数会变得很大，一方面可能
单机内存无法存放导致无法训练，另一方面普通的训练模式每一轮迭代都需要同步完整的参数，参数太大会让通信变得
非常慢，进而影响训练速度。

Fluid支持千亿量级超大规模稀疏特征embedding的训练，embedding参数只会保存在parameter server上，通过
参数prefetch和梯度稀疏更新的方法，大大减少通信量，提高通信速度。

该功能只对分布式训练有效，单机无法使用。
需要配合稀疏更新一起使用。

使用方法，在配置embedding的时候，加上参数 :code:`is_distributed=True` 以及 :code:`is_sparse=True` 即可。
参数 :code:`dict_size` 定义数据中总的id的数量，id可以是int64范围内的任意值，只要总id个数小于等于dict_size就可以支持。
所以配置之前需要预估一下数据中总的feature id的数量。

.. code-block:: python

  emb = fluid.layers.embedding(
      is_distributed=True,
      input=input,
      size=[dict_size, embedding_width],
      is_sparse=True,
      is_distributed=True)


模型存储和预测
=============

当特征数量达到千亿的时候，参数量很大，单机已经无法存下，所以模型的存储和加载都和普通模式不太一样，普通模式下，参数是
在trainer端保存和加载的，而对于分布式embedding的参数，参数的保存和加载，都是在pserver端进行，每个pserver只保存和加载
他自己对应的那部分参数。