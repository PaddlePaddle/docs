.. _cn_api_paddle_cuda_Stream:

Stream
------

.. py:class:: paddle.cuda.Stream(device=None, priority=2, stream_base=None)

CUDA 流类，用于管理异步操作。

参数
::::::::::::
    - **device** (int|paddle.Place|str|int|None，可选) - 设备 ID 或设备对象
    - **priority** (int, 可选) - 流的优先级，可以是 1 或 -1（高优先级）或 0 或 2（低优先级）。默认值为 2。
    - **stream_base** (_InitStreamBase|None，可选) - 用于初始化流的底层 StreamBase 对象。默认值为 None。

代码示例
::::::::::::
COPY-FROM: paddle.cuda.Stream
