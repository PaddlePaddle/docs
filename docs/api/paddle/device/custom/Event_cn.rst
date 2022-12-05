.. _cn_api_device_custom_Event:

Event
-------------------------------

.. py:class:: paddle.device.custom.Event(device, device_id=None, enable_timing=False, blocking=False, interprocess=False)

custom device event 的句柄。

参数
::::::::::::

    - **device** (paddle.CustomPlace()|str) - 希望分配 stream 的设备或设备类型。
    - **device_id** (int，可选) - 希望分配 stream 的设备 ID。
    - **enable_timing** (bool，可选) - Event 是否需要统计时间。默认值为 False。
    - **blocking** (bool，可选) - wait()函数是否被阻塞。默认值为 False。
    - **interprocess** (bool，可选) - Event 是否能在进程间共享。默认值为 False。

返回
::::::::::::
None

代码示例
::::::::::::

COPY-FROM: paddle.device.custom.Event


方法
::::::::::::
record(stream=None)
'''''''''

记录 event 到给定的 stream。

**参数**

    - **stream** (CustomDeviceStream，可选) - custom device stream 的句柄。如果为 None，stream 为当前的 stream。默认值为 None。

**代码示例**

COPY-FROM: paddle.device.custom.Event.record

query()
'''''''''

查询 event 的状态。

**返回**

 一个 boolean 变量，用于标识当前 event 获取的所有任务是否被完成。

**代码示例**

COPY-FROM: paddle.device.custom.Event.query


synchronize()
'''''''''

等待当前 event 完成。

**代码示例**

COPY-FROM: paddle.device.custom.Event.synchronize
