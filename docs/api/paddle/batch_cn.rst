.. _cn_api_paddle_batch:

batch
-------------------------------

.. py:function:: paddle.batch(reader, batch_size, drop_last=False)

一个 reader 的装饰器。返回的 reader 将输入 reader 的数据打包成指定的 batch_size 大小的批处理数据（batched data）。

.. warning::
    不推荐使用这个 API，如有数据加载需求推荐使用支持多进程并发加速的 ``paddle.io.DataLoader``

参数
::::::::::::

    - **reader** (generator)- 读取数据的数据 reader。
    - **batch_size** (int)- 批尺寸。
    - **drop_last** (bool) - 若设置为 True，则当最后一个 batch 不等于 batch_size 时，丢弃最后一个 batch；若设置为 False，则不会。默认值为 False。

返回
::::::::::::
batched reader


代码示例
::::::::::::

COPY-FROM: paddle.batch
