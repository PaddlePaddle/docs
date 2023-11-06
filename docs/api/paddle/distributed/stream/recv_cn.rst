.. _cn_api_paddle_distributed_stream_recv:

recv
-------------------------------


.. py:function:: paddle.distributed.stream.recv(tensor, src=0, group=None, sync_op=True, use_calc_stream=False)

接收一个来自指定进程的 tensor。

参见 :ref:`paddle.distributed.recv<cn_api_paddle_distributed_recv>`。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **tensor** (Tensor) - 用于接收数据的 tensor。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - **src** (int，可选) - 目标进程的 rank，将接收来自该进程的 tensor。默认为 0，即接收来自 rank=0 的进程的 tensor。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **sync_op** (bool，可选) - 该操作是否为同步操作。默认为 True，即同步操作。
    - **use_calc_stream** (bool，可选) - 该操作是否在计算流上进行。默认为 False，即不在计算流上进行。该参数旨在提高同步操作的性能，请确保在充分了解其含义的情况下调整该参数的值。

返回
:::::::::
``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

代码示例
:::::::::
COPY-FROM: paddle.distributed.communication.stream.recv
