.. _cn_api_paddle_distributed_isend:

isend
-------------------------------


.. py:function:: paddle.distributed.isend(tensor, dst, group=None)

异步发送一个 tensor 到指定进程。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **tensor** (Tensor) - 待发送的 Tensor。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - **dst** (int) - 目标进程的 rank，传入的 tensor 将发送到该进程。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。


返回
:::::::::
``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

代码示例
:::::::::
COPY-FROM: paddle.distributed.isend
