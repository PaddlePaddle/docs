.. _cn_api_paddle_io_get_worker_info:

get_worker_info
-------------------------------

.. py:class:: paddle.io.get_worker_info

获取 ``paddle.io.DataLoader`` 子进程信息的函数，用于 ``paddle.io.IterableDataset`` 中划分子进程数据。子进程信息包含以下字段：

``num_workers``：子进程数。

``id``：子进程逻辑序号，从 0 到 ``num_workers - 1``

``dataset``：各子进程中数据集实例。

示例代码见 ``paddle.io.IterableDataset``
