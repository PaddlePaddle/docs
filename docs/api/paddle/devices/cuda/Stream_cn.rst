.. _cn_api_device_cuda_Stream:

Stream
-------------------------------

.. py:class:: paddle.device.cuda.Stream(device=None, priority=None)

CUDA stream的句柄。

参数：
    - **device** (paddle.CUDAPlace()|int|None, 可选) - 希望分配stream的设备。如果是None或者负数，则设备为当前的设备。如果是正数，则必须小于设备的个数。默认值为None。
    - **priority** (int|None, 可选) - stream的优先级。优先级可以为1（高优先级）或者2（正常优先级）。如果优先级为None，优先级为2（正常优先级）。默认值为None。


**代码示例**：

.. code-block:: python

    # required: gpu
    import paddle
    s1 = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
    s2 = paddle.device.cuda.Stream(0, 1)
    s3 = paddle.device.cuda.Stream()



.. py:method:: wait_event(event)

使所有将来提交到stream的任务等待event中已获取的任务。

参数：
    - **event** (CUDAEvent) - 要等待的event。

**代码示例**：

.. code-block:: python

    # required: gpu
    import paddle
    s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
    event = paddle.device.cuda.Event()
    s.wait_event(event)


.. py:method:: wait_stream(stream)

和给定的stream 保持同步。

参数：
    - **stream** (CUDAStream) - 要同步的stream。


**代码示例**：

.. code-block:: python

    # required: gpu
    import paddle
    s1 = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
    s2 = paddle.device.cuda.Stream(0, 1)
    s1.wait_stream(s2)


.. py:method:: query()

返回stream 中所有的操作是否完成的状态。

返回： 一个boolean 值。

**代码示例**：

.. code-block:: python

    # required: gpu
    import paddle
    s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
    is_done = s.query()

.. py:method:: synchronize()

等待所有的stream的任务完成。

**代码示例**：

.. code-block:: python

    # required: gpu
    import paddle
    s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
    s.synchronize()

.. py:method:: record_event(event=None)

标记一个CUDA event 到当前stream中。

参数：
    - **event** (CUDAEvent，可选) - 要标记的event。如果event 为None，新建一个event。默认值为None。

返回： 被标记的event。

**代码示例**：

.. code-block:: python

    # required: gpu
    import paddle
    s = paddle.device.cuda.Stream(paddle.CUDAPlace(0), 1)
    event = s.record_event()

