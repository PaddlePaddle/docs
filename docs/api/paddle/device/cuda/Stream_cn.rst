.. _cn_api_paddle_device_cuda_Stream:

Stream
-------------------------------

.. py:class:: paddle.device.cuda.Stream(device=None, priority=None)

CUDA stream 的句柄。

参数
::::::::::::

    - **device** (paddle.CUDAPlace()|int|None，可选) - 希望分配 stream 的设备。如果是 None 或者负数，则设备为当前的设备。如果是正数，则必须小于设备的个数。默认值为 None。
    - **priority** (int|None，可选) - stream 的优先级。优先级可以为 1（高优先级）或者 2（正常优先级）。如果优先级为 None，优先级为 2（正常优先级）。默认值为 None。


代码示例
::::::::::::

COPY-FROM: paddle.device.cuda.Stream



方法
::::::::::::
wait_event(event)
'''''''''

使所有将来提交到 stream 的任务等待 event 中已获取的任务。

**参数**

    - **event** (CUDAEvent) - 要等待的 event。

**代码示例**

COPY-FROM: paddle.device.cuda.Stream.wait_event


wait_stream(stream)
'''''''''

和给定的 stream 保持同步。

**参数**

    - **stream** (CUDAStream) - 要同步的 stream。


**代码示例**

COPY-FROM: paddle.device.cuda.Stream.wait_stream


query()
'''''''''

返回 stream 中所有的操作是否完成的状态。

**返回**
 一个 boolean 值。

**代码示例**

COPY-FROM: paddle.device.cuda.Stream.query

synchronize()
'''''''''

等待所有的 stream 的任务完成。

**代码示例**

COPY-FROM: paddle.device.cuda.Stream.synchronize

record_event(event=None)
'''''''''

标记一个 CUDA event 到当前 stream 中。

**参数**

    - **event** (CUDAEvent，可选) - 要标记的 event。如果 event 为 None，新建一个 event。默认值为 None。

**返回**
 被标记的 event。

**代码示例**

COPY-FROM: paddle.device.cuda.Stream.record_event
