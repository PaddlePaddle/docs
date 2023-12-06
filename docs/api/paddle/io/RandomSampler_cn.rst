.. _cn_api_paddle_io_RandomSampler:

RandomSampler
-------------------------------

.. py:class:: paddle.io.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)

随机迭代样本，产生重排下标，如果 ``replacement = False``，则会采样整个数据集；如果 ``replacement = True``，则会按照 ``num_samples`` 指定的样本数采集。

参数
:::::::::
    - **data_source** (Dataset) - 此参数必须是 :ref:`cn_api_paddle_io_Dataset` 或 :ref:`cn_api_paddle_io_IterableDataset` 的一个子类实例或实现了 ``__len__`` 的 Python 对象，用于生成样本下标。默认值为 None。
    - **replacement** (bool，可选) - 如果为 ``False`` 则会采样整个数据集，如果为 ``True`` 则会按 ``num_samples`` 指定的样本数采集。默认值为 ``False`` 。
    - **num_samples** (int，可选) - 如果 ``replacement`` 设置为 ``True`` 则按此参数采集对应的样本数。默认值为 None,不启用。
    - **generator** (Generator，可选) - 指定采样 ``data_source`` 的采样器。默认值为 None，不启用。

返回
:::::::::
RandomSampler，返回随机采样下标的采样器


代码示例
:::::::::

COPY-FROM: paddle.io.RandomSampler
