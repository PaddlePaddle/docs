.. _cn_api_paddle_utils_data_DataLoader:

DataLoader
-------------------------------

.. py:class:: paddle.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device='', in_order=True)

按批次加载数据集的 PyTorch 兼容接口，基于 :ref:`cn_api_paddle_io_DataLoader` 实现。

参数
:::::::::

    - **dataset** (Dataset) - 要加载的数据集。
    - **batch_size** (int|None，可选) - 每个批次的样本数。默认值：1。
    - **shuffle** (bool，可选) - 是否在每个 epoch 打乱数据。默认值：False。
    - **sampler** (BatchSampler|None，可选) - 兼容参数，作为批次索引采样器使用，不能与 ``batch_sampler`` 同时设置。默认值：None。
    - **batch_sampler** (BatchSampler|None，可选) - 指定批次索引的采样器。默认值：None。
    - **num_workers** (int，可选) - 加载数据的子进程数。默认值：0。
    - **collate_fn** (Callable|None，可选) - 将样本列表整理成批次的函数。默认值：None。
    - **pin_memory** (bool，可选) - 兼容保留参数；当前实现会忽略 True。默认值：False。
    - **drop_last** (bool，可选) - 是否丢弃最后一个不完整批次。默认值：False。
    - **timeout** (float，可选) - 从工作进程获取批次的超时时间，单位为秒。默认值：0。
    - **worker_init_fn** (Callable|None，可选) - 每个工作进程启动后调用的初始化函数。默认值：None。
    - **multiprocessing_context** (Any|None，可选) - 兼容保留参数，当前实现会忽略非 None 值。默认值：None。
    - **generator** (Any|None，可选) - 兼容保留参数，当前实现会忽略非 None 值。默认值：None。

关键字参数
:::::::::::

    - **prefetch_factor** (int|None，可选) - 兼容保留参数，当前实现会忽略非 None 值。默认值：None。
    - **persistent_workers** (bool，可选) - 是否在数据集消费完一次后保持工作进程存活。默认值：False。
    - **pin_memory_device** (str，可选) - 兼容保留参数，当前实现会忽略非空值。默认值：''。
    - **in_order** (bool，可选) - 兼容保留参数，当前实现会忽略 False。默认值：True。

返回
:::::::::

DataLoader，可迭代的数据加载器。
