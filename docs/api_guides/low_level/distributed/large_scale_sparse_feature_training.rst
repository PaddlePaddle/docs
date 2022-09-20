.. _api_guide_large_scale_sparse_feature_training:

###################
大规模稀疏特征模型训练
###################


模型配置和训练
=============

embedding 被广泛应用在各种网络结构中，尤其是文本处理相关的模型。在某些场景，例如推荐系统或者搜索引擎中，
embedding 的 feature id 可能会非常多，当 feature id 达到一定数量时，embedding 参数会变得很大，
会带来两个问题：

1. 单机内存由于无法存放如此巨大的 embedding 参数，导致无法训练；
2. 普通的训练模式每一轮迭代都需要同步完整的参数，参数太大会让通信变得非常慢，进而影响训练速度。

Fluid 支持千亿量级超大规模稀疏特征 embedding 的训练，embedding 参数只会保存在 parameter server 上，通过
参数 prefetch 和梯度稀疏更新的方法，大大减少通信量，提高通信速度。

该功能只对分布式训练有效，单机无法使用。
需要配合 `稀疏更新 <../layers/sparse_update.html>`_ 一起使用。

使用方法：在配置 embedding 的时候，加上参数 :code:`is_distributed=True` 以及 :code:`is_sparse=True` 即可。
参数 :code:`dict_size` 定义数据中总的 id 的数量，id 可以是 int64 范围内的任意值，只要总 id 个数小于等于 dict_size 就可以支持。
所以配置之前需要预估一下数据中总的 feature id 的数量。

.. code-block:: python

  emb = fluid.layers.embedding(
      is_distributed=True,
      input=input,
      size=[dict_size, embedding_width],
      is_sparse=True,
      is_distributed=True)


模型存储和预测
=============

当特征数量达到千亿的时候，参数量很大，单机已经无法存下，所以模型的存储和加载都和普通模式不同：

1. 普通模式下，参数是在 trainer 端保存和加载的；
2. 分布式模式下，参数的保存和加载，都是在 pserver 端进行，每个 pserver 只保存和加载该 pserver 自身对应部分的参数
