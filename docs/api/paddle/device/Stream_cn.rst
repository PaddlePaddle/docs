.. _cn_api_paddle_device_Stream:

Stream
-------------------------------

.. py:class:: paddle.device.Stream(device=None, priority=2, stream_base=None)

设备 stream 的句柄。``paddle.cuda.Stream()`` 与 ``paddle.device.Stream()`` 等价。

参数
::::::::::::

    - **device** (paddle.CUDAPlace|paddle.CustomPlace|str) - 希望分配 stream 的设备或设备类型。如果为 None，则为当前期望的 place。默认值为 None。
    - **priority** (int，可选) - 流的优先级，可以是 1 或 -1（高优先级）或 0 或 2（低优先级）。默认值为 2。
    - **stream_base** (_InitStreamBase|None，可选) - 用于初始化流的底层 StreamBase 对象。默认值为 None。


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

query()
'''''''''

返回 stream 中所有的操作是否完成的状态。

**返回**
 一个 boolean 值。

**代码示例**

COPY-FROM: paddle.device.Stream.query

synchronize()
'''''''''

等待当前 stream 中所有 kernel 完成。

**代码示例**

COPY-FROM: paddle.device.Stream.synchronize
