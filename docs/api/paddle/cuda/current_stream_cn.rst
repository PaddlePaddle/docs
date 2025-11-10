.. _cn_api_paddle_cuda_current_stream:

current_stream
-------------------------------

.. py:function:: paddle.cuda.current_stream(device=None)

该功能用于获取指定 CUDA 设备上当前正在使用的计算流（Stream）。

参数
:::::::::
    - **device** (int | str | CUDAPlace | CustomPlace | None，可选) – 指定需要查询的设备。
        - None：获取当前设备上的默认计算流。
        - int：设备索引，例如 0 表示 cuda:0。
        - str：设备字符串，例如 'cuda:0' 或 'gpu:1'。
        - CUDAPlace：Paddle 的 CUDAPlace 对象。
        - CustomPlace：Paddle 的自定义设备 Place 对象。

返回
:::::::::
core.CUDAStream，当前设备对应的 CUDA 计算流对象。

代码示例
:::::::::
COPY-FROM: paddle.cuda.current_stream
