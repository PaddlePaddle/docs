.. _cn_api_paddle_cuda_is_current_stream_capturing:

is_current_stream_capturing
-------------------------------

.. py:function:: paddle.cuda.is_current_stream_capturing()

检查当前设备流是否处于图形捕获状态。

返回
:::::::::
bool: 如果当前流正在捕获则返回True，否则返回False。

代码示例
::::::::::::
COPY-FROM: paddle.cuda.is_current_stream_capturing