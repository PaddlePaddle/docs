.. _cn_api_paddle_distributed_shard_dataloader:

shard_dataloader
-------------------------------

.. py:function:: paddle.distributed.shard_dataloader(dataloader, meshes, input_keys=None, shard_dims=None, is_dataset_splitted=False)

将单卡视角的数据加载器转变为分布式视角，与普通的 dataloader 相比，其提供了两个能力：
1. 如果 dataloader 的 shard_dim 不为 None，则按 shard_dim 拆分 dataloader 以进行数据并行。
2. 将 dataloader 的输出添加上分布式属性，即把普通 tensor 变为分布式 tensor。


参数
:::::::::

    - **dataloader** (paddle.io.DataLoader) - 单卡视角的 dataloader。
    - **meshes** (ProcessMesh|list|tuple) - 切分 dataloader 使用的 mesh。可以是个 ProcessMesh 或者 list，如果是个 list，则表示不同的输入需要在不同的 mesh 上。
    - **input_keys** (list|tuple，可选) - 如果 dataloader 的迭代结果是一个张量字典，input_keys 是这个字典的键，标识哪个张量位于哪个 mesh 上，与 meshes 一一对应。默认值 None，表示 dataloader 的迭代结果不是 dict。
    - **shard_dims** (str|int|list|tuple，可选) - 对 dataloader 进行分片的 mesh 维度。默认值 None，代表不切分 dataloader，通常使用数据并行的情况下，必须设置此参数。
    - **is_dataset_splitted** (bool，可选) - 数据集是否已根据数据并行的 rank 进行了切分。默认值 False。

返回
:::::::::
ShardDataloader：一个具有分布式视角的数据加载器对象。


代码示例
:::::::::

COPY-FROM: paddle.distributed.shard_dataloader
