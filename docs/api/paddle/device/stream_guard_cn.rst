.. _cn_api_paddle_device_stream_guard:

stream_guard
-------------------------------

.. py:function:: paddle.device.stream_guard(stream)

可以切换当前的 stream 为输入指定的 stream。

.. note::
    该 API 目前仅支持动态图模式。

参数
::::::::::::

    - **stream** (paddle.device.Stream) - 指定的 stream。如果为 None，则不进行 stream 切换。

代码示例
::::::::::::
COPY-FROM: paddle.device.stream_guard
