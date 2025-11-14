.. _cn_api_paddle_cuda_get_stream_from_external:

get_stream_from_external
------------------------

.. py:function:: paddle.cuda.get_stream_from_external(data_ptr, device=None)

从外部创建的 CUDA 流创建 Paddle 流对象。

参数
::::::::::::
    - **data_ptr** (int) - 外部 CUDA 流的指针值
    - **device** (int, 可选) - 设备 ID，默认为 None

返回
::::::::::::
    paddle.cuda.Stream: 包装后的 Paddle 流对象

代码示例
::::::::::::
COPY-FROM: paddle.cuda.get_stream_from_external
