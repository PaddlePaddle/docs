.. _cn_api_paddle_io_IterableDataset:

IterableDataset
-------------------------------

.. py:class:: paddle.io.IterableDataset

概述迭代式数据集的方法和行为的抽象类。

迭代式(iterable style)数据集需要继承这个基类，迭代式数据集为只能依次迭代式获取样本的数据集，类似 Python 中的迭代器，所有迭代式数据集须实现以下方法：

``__iter__``：依次返回数据赝本。

.. note::
    迭代式数据集不需要实现 ``__getitem__`` 和 ``__len__``，也不可以调用迭代式数据集的这两个方法。

见 :ref:`cn_api_paddle_io_DataLoader` 。

代码示例 1
::::::::::::

COPY-FROM: paddle.io.IterableDataset:code-example1

当 ``paddle.io.DataLoader`` 中 ``num_workers > 0`` 时，每个子进程都会遍历全量的数据集返回全量样本，所以数据集会重复 ``num_workers``
次，如果需要数据集样本不会重复返回，可通过如下两种方法避免样本重复，两种方法中都需要通过 ``paddle.io.get_worker_info`` 获取各子进程的信息。


代码示例 2
::::::::::::

通过 ``__iter__`` 函数划分各子进程的数据

COPY-FROM: paddle.io.IterableDataset:code-example2


代码示例 3
::::::::::::

通过各子进程初始化函数 ``worker_inif_fn`` 划分子进程数据

COPY-FROM: paddle.io.IterableDataset:code-example3
