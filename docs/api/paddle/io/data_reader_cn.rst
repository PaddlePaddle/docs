.. _cn_api_io_data_reader:

data_reader
-------------------------------

.. py:class:: paddle.io.data_reader(reader_func, batch_size=1, num_samples=1, shuffle=False, drop_last=False, seed=None)

用于在GPU DataLoader流水线中启动数据集读取阶段，此阶段会通过独立的子线程来执行

参数
::::::::::::

    - **reader_func** (callable) - 定义阶段内数据集读取的函数。
    - **batch_size** (int, 可选) - 每个批次的样本数，默认为1。
    - **num_samples** (int, 可选) - 总共读取数据集的样本数，默认为1。
    - **shuffle** (bool, 可选) - 是否打乱数据集读取顺序，默认为False。
    - **drop_last** (bool, 可选) - 是否丢弃因数据集样本数不能被 ``batch_size`` 整除而产生的最后一个不完整的批次，默认为False。
    - **seed** (int, 可选) - 打乱数据集读取顺序时使用的随机种子，默认为None。

返回
::::::::::::
    数据集读取函数的输出


代码示例
::::::::::::

COPY-FROM: paddle.io.data_reader:code-example
