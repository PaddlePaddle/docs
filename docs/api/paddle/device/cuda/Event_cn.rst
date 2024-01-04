.. _cn_api_paddle_device_cuda_Event:

Event
-------------------------------

.. py:class:: paddle.device.cuda.Event(enable_timing=False, blocking=False, interprocess=False)

CUDA event 的句柄。

参数
::::::::::::

    - **enable_timing** (bool，可选) - Event 是否需要统计时间。默认值为 False。
    - **blocking** (bool，可选) - wait()函数是否被阻塞。默认值为 False。
    - **interprocess** (bool，可选) - Event 是否能在进程间共享。默认值为 False。

返回
::::::::::::
None

代码示例
::::::::::::

COPY-FROM: paddle.device.cuda.Event


.. warning::
    该 API 未来计划废弃，不推荐使用。
