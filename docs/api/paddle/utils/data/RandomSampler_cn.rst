.. _cn_api_paddle_utils_data_RandomSampler:

RandomSampler
-------------------------------

.. py:class:: paddle.utils.data.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)

``paddle.io.RandomSampler`` 的别名，请参考 :ref:`cn_api_paddle_io_RandomSampler`。

参数
:::::::::

    - **data_source** (Dataset) - 必填，要进行随机采样的数据集。
    - **replacement** (bool，可选) - 是否有放回采样。默认值：False。
    - **num_samples** (int|None，可选) - 要采样的样本数。默认值：None。
    - **generator** (Generator|None，可选) - 随机数生成器。默认值：None。
