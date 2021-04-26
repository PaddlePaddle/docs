.. _cn_api_io_cn_random_split:

random_split
-------------------------------

.. py:class:: paddle.io.random_split(dataset, lengths, generator=None)

给定子集合dataset的长度数组，随机切分出原数据集合的非重复子集合。

参数:
    - **dataset** (Dataset) - 此参数必须是 ``paddle.io.Dataset`` 或 ``paddle.io.IterableDataset`` 的一个子类实例或实现了 ``__len__`` 的Python对象，用于生成样本下标。默认值为None。
    - **lengths** (list) - 总和为原数组长度的，子集合长度数组。
    - **generator** (Generator) - 指定采样 ``data_source`` 的采样器。默认值为None。

返回: list, 返回按给定长度数组描述随机分割的原数据集合的非重复子集合。


**代码示例**

.. code-block:: python

    import paddle
    from paddle.io import random_split
    a_list = paddle.io.random_split(range(10), [3, 7])
    print(len(a_list)) 
    # 2
    for idx, v in enumerate(a_list[0]):
        print(idx, v)
    # output of the first subset
    # 0 1
    # 1 3
    # 2 9
    for idx, v in enumerate(a_list[1]):
        print(idx, v)
    # output of the second subset
    # 0 5
    # 1 7
    # 2 8
    # 3 6
    # 4 0
    # 5 2
    # 6 4
