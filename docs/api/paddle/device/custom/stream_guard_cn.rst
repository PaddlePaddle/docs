.. _cp_api_device_custom_stream_guard:

stream_guard
-------------------------------

.. py:function:: paddle.device.custom.stream_guard(stream)

可以切换当前的 custom device stream 为输入指定的 stream。

.. note::
    该 API 目前仅支持动态图模式。

参数
::::::::::::

    - **stream** (paddle.device.custom.Stream) - 指定的 custom device stream。如果为 None，则不进行 stream 流切换。

代码示例
::::::::::::
COPY-FROM: paddle.device.custom.stream_guard
