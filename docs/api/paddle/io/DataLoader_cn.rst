.. _cn_api_paddle_io_DataLoader:

DataLoader
-------------------------------

.. py:class:: paddle.io.DataLoader(dataset, feed_list=None, places=None, return_list=False, batch_sampler=None, batch_size=1, shuffle=False, drop_last=False, collate_fn=None, num_workers=0, use_buffer_reader=True, use_shared_memory=True, prefetch_factor=2, timeout=0, worker_init_fn=None)

DataLoader 返回一个迭代器，该迭代器根据 ``batch_sampler`` 给定的顺序迭代一次给定的 ``dataset``

DataLoader 支持单进程和多进程的数据加载方式，当 ``num_workers`` 大于 0 时，将使用多进程方式异步加载数据。

DataLoader 当前支持 ``map-style`` 和 ``iterable-style`` 的数据集，``map-style`` 的数据集可通过下标索引样本，请参考 ``paddle.io.Dataset`` ； ``iterable-style`` 数据集只能迭代式地获取样本，类似 Python 迭代器，请参考 ``paddle.io.IterableDataset`` 。

.. note::

    当前还不支持在子进程中进行 GPU Tensor 的操作，请不要在子进程流程中使用 GPU Tensor，例如 ``dataset`` 中的预处理，``collate_fn`` 等，``numpy array`` 和 CPU Tensor 操作已支持。

``batch_sampler`` 请参考 ``paddle.io.BatchSampler``

**禁用自动组 batch**

在如 NLP 等任务中，用户需求自定义组 batch 的方式，不希望 ``DataLoader`` 自动组 batch， ``DataLoader`` 支持在 ``batch_size`` 和 ``batch_sampler`` 均为 None 的时候禁用自动组 batch 功能，此时需求从 ``dataset`` 中获取的数据为已经组好 batch 的数据，该数据将不做任何处理直接传到 ``collate_fn`` 或 ``default_collate_fn`` 中。

.. note::

    当禁用自动组 batch 时，``default_collate_fn`` 将不对输入数据做任何处理。

参数
::::::::::::

    - **dataset** (Dataset) - DataLoader 从此参数给定数据集中加载数据，此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例。
    - **feed_list** (list(Tensor)|tuple(Tensor)，可选) - feed 变量列表，由 ``paddle.static.data()`` 创建。当 ``return_list`` 为 False 时，此参数必须设置。默认值为 None。
    - **places** (list(Place)|tuple(Place)，可选) - 数据需要放置到的 Place 列表。在静态图和动态图模式中，此参数均必须设置。在动态图模式中，此参数列表长度必须是 1。默认值为 None。
    - **return_list** (bool，可选) - 每个设备上的数据是否以 list 形式返回。若 return_list = False，每个设备上的返回数据均是 str -> Tensor 的映射表，其中映射表的 key 是每个输入变量的名称。若 return_list = True，则每个设备上的返回数据均是 list(Tensor)。在动态图模式下，此参数必须为 True。默认值为 False。
    - **batch_sampler** (BatchSampler，可选) - ``paddle.io.BatchSampler`` 或其子类的实例，DataLoader 通过 ``batch_sampler`` 产生的 mini-batch 索引列表来 ``dataset`` 中索引样本并组成 mini-batch。默认值为 None。
    - **batch_size** (int|None，可选) - 每 mini-batch 中样本个数，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``paddle.io.BatchSampler``。默认值为 1。
    - **shuffle** (bool，可选) - 生成 mini-batch 索引列表时是否对索引打乱顺序，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``paddle.io.BatchSampler``。默认值为 False。
    - **drop_last** (bool，可选) - 是否丢弃因数据集样本数不能被 ``batch_size`` 整除而产生的最后一个不完整的 mini-batch，为 ``batch_sampler`` 的替代参数，若 ``batch_sampler`` 未设置，会根据 ``batch_size`` ``shuffle`` ``drop_last`` 创建一个 ``paddle.io.BatchSampler``。默认值为 False。
    - **collate_fn** (callable，可选) - 通过此参数指定如何将样本列表组合为 mini-batch 数据，当 ``collate_fn`` 为 None 时，默认为将样本个字段在第 0 维上堆叠(同 ``np.stack(..., axis=0)`` )为 mini-batch 的数据。默认值为 None。
    - **num_workers** (int，可选) - 用于加载数据的子进程个数，若为 0 即为不开启子进程，在主进程中进行数据加载。默认值为 0。
    - **use_buffer_reader** (bool，可选) - 是否使用缓存读取器。若 ``use_buffer_reader`` 为 True，DataLoader 会异步地预读取一定数量（默认读取下一个）的 mini-batch 的数据，可加速数据读取过程，但同时会占用少量的 CPU/GPU 存储，即一个 batch 输入数据的存储空间。默认值为 True。
    - **prefetch_factor** (int，可选) - 缓存的 mini-batch 的个数。若 ``use_buffer_reader`` 为 True，DataLoader 会异步地预读取 ``prefetch_factor`` 个 mini-batch。默认值为 2。
    - **use_shared_memory** (bool，可选) - 是否使用共享内存来提升子进程将数据放入进程间队列的速度，该参数仅在多进程模式下有效(即 ``num_workers > 0`` )，请确认机器上有足够的共享内存空间(如 Linux 系统下 ``/dev/shm/`` 目录空间大小)再设置此参数。默认为 True。
    - **timeout** (int，可选) - 从子进程输出队列获取 mini-batch 数据的超时时间。默认值为 0。
    - **worker_init_fn** (callable，可选) - 子进程初始化函数，此函数会被子进程初始化时被调用，并传递 ``worker id`` 作为参数。默认值为 None。

返回
::::::::::::
DataLoader，迭代 ``dataset`` 数据的迭代器，迭代器返回的数据中的每个元素都是一个 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.io.DataLoader
