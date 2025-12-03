.. _cn_api_paddle_cuda_synchronize:

synchronize

.. py:function:: paddle.cuda.synchronize(device=None)

该功能用于同步指定 CUDA 设备上的计算流，确保所有在该设备上提交的计算任务执行完成。

参数
:::::::::
    - **device** (int | str | CUDAPlace | CustomPlace | None, optional) – 指定需要同步的设备。
        - None：同步当前设备上的计算流。
        - int：设备索引，例如 0 表示 cuda:0。
        - str：设备字符串，例如 'cuda:0' 或 'gpu:0'。
        - CUDAPlace：Paddle 的 CUDAPlace 对象。
        - CustomPlace：Paddle 的自定义设备 Place 对象。

返回
:::::::::
无。

代码示例
:::::::::
COPY-FROM: paddle.cuda.synchronize
