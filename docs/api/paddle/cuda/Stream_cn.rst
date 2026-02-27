.. _cn_api_paddle_cuda_Stream:

Stream
------

.. py:class:: paddle.cuda.Stream(device=None, priority=0, blocking=False)

CUDA 流类，用于管理异步操作。

参数
::::::::::::
    - **device** (int|paddle.Place|str|int|None) - 设备 ID 或设备对象
    - **priority** (int, 可选) - 流的优先级，默认为 None; 可以是 1 或-1（高优先级）或 0 或 2（低优先级）。默认情况下，流具有优先级 0。
    - **blocking** (bool|None，可选) - stream 是否同步执行。默认值为 False。

代码示例
::::::::::::
COPY-FROM: paddle.cuda.Stream
