.. _cn_api_paddle_utils_data_BatchSampler:

BatchSampler
-------------------------------

.. py:class:: paddle.utils.data.BatchSampler(dataset=None, sampler=None, shuffle=False, batch_size=1, drop_last=False)

``paddle.io.BatchSampler`` 的别名，也支持 PyTorch 风格签名 ``paddle.utils.data.BatchSampler(sampler=None, batch_size=1, drop_last=False)``。请参考 :ref:`cn_api_paddle_io_BatchSampler`。

参数
:::::::::

    - **dataset** (Dataset|None，可选) - 用于生成样本索引的数据集。默认值：None。
    - **sampler** (Sampler|Iterable|None，可选) - 样本索引采样器，不能与 ``dataset`` 同时设置。默认值：None。
    - **shuffle** (bool，可选) - 是否打乱索引。默认值：False。
    - **batch_size** (int，可选) - 每个批次的样本数。默认值：1。
    - **drop_last** (bool，可选) - 是否丢弃最后一个不完整批次。默认值：False。
