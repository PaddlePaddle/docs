.. _cn_api_paddle_utils_data_distributed_DistributedSampler:

DistributedSampler
-------------------------------

.. py:class:: paddle.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)

为分布式训练划分数据集，使每个进程获得互斥的数据子集。

参数
:::::::::

    - **dataset** (Sized) - 要采样的数据集，必须实现 ``__len__``。
    - **num_replicas** (int|None，可选) - 参与训练的进程数。默认值：None，使用当前分布式环境的进程数。
    - **rank** (int|None，可选) - 当前进程的 rank。默认值：None，使用当前分布式环境的 rank。
    - **shuffle** (bool，可选) - 是否按 ``seed`` 在每个 epoch 打乱索引。默认值：True。
    - **seed** (int，可选) - 打乱索引使用的随机种子。默认值：0。
    - **drop_last** (bool，可选) - 是否丢弃尾部数据，使数据能被各进程均分。默认值：False。

返回
:::::::::

DistributedSampler，分布式样本索引迭代器。
