.. _cn_api_io_cn_DistributedBatchSampler:

DistributedBatchSampler
-------------------------------

.. py:class:: paddle.io.DistributedBatchSampler(dataset, batch_size, num_replicas=None, rank=None, shuffle=False, drop_last=False)

分布式批采样器加载数据的一个子集。每个进程可以传递给 DataLoader 一个 DistributedBatchSampler 的实例，每个进程加载原始数据的一个子集。


.. note::
  假定 Dataset 的大小是固定的。

参数
::::::::::::

    - **dataset** (Dataset) - 此参数必须是 :ref:`cn_api_io_cn_Dataset` 的一个子类实例或实现了 ``__len__`` 的 Python 对象，用于生成样本下标。
    - **batch_size** (int) - 每 mini-batch 中包含的样本数。
    - **num_replicas** (int，可选) - 分布式训练时的进程个数。如果是 None，会依据 :ref:`cn_api_fluid_dygraph_ParallelEnv` 获取值。默认是 None。
    - **rank** (int，可选) - num_replicas 个进程中的进程序号。如果是 None，会依据 :ref:`cn_api_fluid_dygraph_ParallelEnv` 获取值。默认是 None。
    - **shuffle** (bool，可选) - 是否需要在生成样本下标时打乱顺序。默认值为 False。
    - **drop_last** (bool，可选) - 是否需要丢弃最后无法凑整一个 mini-batch 的样本。默认值为 False。


返回
::::::::::::
DistributedBatchSampler，返回样本下标数组的迭代器。


代码示例
::::::::::::

COPY-FROM: paddle.io.DistributedBatchSampler

方法
::::::::::::
set_epoch(epoch)
'''''''''

设置 epoch 数。当设置``shuffle=True``时，此 epoch 被用作随机种子。默认情况下，用户可以不用此接口设置，每个 epoch 时，所有的进程(workers)使用不同的顺序。如果每个 epoch 设置相同的数字，每个 epoch 数据的读取顺序将会相同。

**参数**

    - **epoch** (int) - epoch 数。

**代码示例**

COPY-FROM: paddle.io.DistributedBatchSampler.set_epoch
