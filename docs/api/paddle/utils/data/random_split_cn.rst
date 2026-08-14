.. _cn_api_paddle_utils_data_random_split:

random_split
-------------------------------

.. py:function:: paddle.utils.data.random_split(dataset, lengths, generator=None)

``paddle.io.random_split`` 的别名。将数据集随机划分为给定长度的互不重叠的新数据集。

参数
:::::::::

    - **dataset** (Dataset) - 要划分的数据集。
    - **lengths** (Sequence) - 各划分的长度或比例。
    - **generator** (Generator，可选) - 用于随机排列的生成器。默认值为 None，此时使用 ``manual_seed()`` 中的默认生成器。

返回
:::::::::

list[Subset]，原始数据集的互不重叠子数据集列表。
