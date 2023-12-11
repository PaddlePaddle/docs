.. _cn_api_paddle_io_ConcatDataset:

ConcatDataset
-------------------------------

.. py:class:: paddle.io.ConcatDataset(datasets)

将多个数据集拼接为一个。

此 API 可用于集成多个不同的数据集。

参数
:::::::::

    - **datasets** (sequence) - 待拼接的数据集序列。

返回
:::::::::

Dataset，由 ``datasets`` 拼接而成的数据集。

代码示例
:::::::::

COPY-FROM: paddle.io.ConcatDataset
