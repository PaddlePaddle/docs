.. _cn_api_paddle_device_Stream:

Stream
-------------------------------

.. py:class:: paddle.device.cuda.Stream(device=None, priority=None, blocking=False)

custom device stream 的句柄。

参数
::::::::::::

    - **device** (paddle.CUDAPlace|paddle.CustomPlace|str) - 希望分配 stream 的设备或设备类型。如果为 None，则为当前期望的 place。默认值为 None。
    - **priority** (int|None，可选) - stream 的优先级。优先级可以为 1（高优先级）或者 2（正常优先级）。如果优先级为 None，优先级为 2（正常优先级）。默认值为 None。
    - **blocking** (bool|None，可选) - stream 是否同步执行。默认值为 False。


代码示例
::::::::::::

COPY-FROM: paddle.device.Stream

方法
::::::::::::

record_event(event=None)
'''''''''

标记一个 event 到当前 stream 中。

**参数**

    - **event** (paddle.device.Event) - 要标记的 event。如果 event 为 None，新建一个 event。默认值为 None。

**返回**
 被标记的 event。

**代码示例**

COPY-FROM: paddle.device.Stream.record_event

wait_event(event)
'''''''''

使所有将来提交到 stream 的任务等待 event 中已获取的任务。

**参数**

    - **event** (paddle.device.Event) - 要等待的 event。

**代码示例**

COPY-FROM: paddle.device.Stream.wait_event


wait_stream(stream)
'''''''''

和给定的 stream 保持同步。

**参数**

    - **stream** (paddle.device.Stream) - 要同步的 stream。


**代码示例**

COPY-FROM: paddle.device.Stream.wait_stream

record_event(event=None)
'''''''''

记录给定的 event。

**参数**

    - **event** (paddle.device.Event) - 要记录的 event，如果为 None，则新建一个 event。


**代码示例**

COPY-FROM: paddle.device.Stream.record_event

query()
'''''''''

返回 stream 中所有的操作是否完成的状态。

**返回**
 一个 boolean 值。

**代码示例**

COPY-FROM: paddle.device.Stream.query

synchronize()
'''''''''

等待所有的 stream 的任务完成。

**代码示例**

COPY-FROM: paddle.device.Stream.synchronize
