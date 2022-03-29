.. _cn_api_device_cuda_Event:

Event
-------------------------------

.. py:class:: paddle.device.cuda.Event(enable_timing=False, blocking=False, interprocess=False)

CUDA event的句柄。

参数
::::::::::::

    - **enable_timing** (bool, 可选) - Event 是否需要统计时间。默认值为False。
    - **blocking** (bool, 可选) - wait()函数是否被阻塞。默认值为False。
    - **interprocess** (bool, 可选) - Event是否能在进程间共享。默认值为False。

返回
::::::::::::
None

代码示例
::::::::::::

.. code-block:: python

    # required: gpu
    import paddle
    event = paddle.device.cuda.Event()


方法
::::::::::::
record(CUDAStream=None)
'''''''''

记录event 到给定的stream。

参数
::::::::::::

    - **stream** (CUDAStream, 可选) - CUDA stream的句柄。如果为None，stream为当前的stream。默认值为False。

代码示例
::::::::::::

.. code-block:: python

    # required: gpu
    import paddle
    event = paddle.device.cuda.Event()
    event.record()

query()
'''''''''

查询event的状态。

返回
::::::::::::
 一个boolean 变量，用于标识当前event 获取的所有任务是否被完成。

代码示例
::::::::::::

.. code-block:: python

    # required: gpu
    import paddle
    event = paddle.device.cuda.Event()
    is_done = event.query()


synchronize()
'''''''''

等待当前event 完成。

代码示例
::::::::::::

.. code-block:: python

    # required: gpu
    import paddle
    event = paddle.device.cuda.Event()
    event.synchronize()
