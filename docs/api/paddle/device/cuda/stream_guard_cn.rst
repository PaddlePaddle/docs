.. _cp_api_device_cuda_stream_guard:

stream_guard
-------------------------------

.. py:function:: paddle.device.cuda.stream_guard(stream)

可以切换当前的CUDA stream为输入指定的stream。


参数：
    - **stream** (paddle.device.cuda.Stream) - 指定的CUDA stream。如果为None，则不进行stream流切换。

**代码示例**：
COPY-FROM: paddle.device.cuda.stream_guard
