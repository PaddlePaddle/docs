.. _cn_api_paddle_distributed_stream_reduce:

reduce
-------------------------------


.. py:function:: paddle.distributed.stream.reduce(tensor, dst=0, op=ReduceOp.SUM, group=None, sync_op=True, use_calc_stream=False)

规约进程组内的一个 tensor，随后将结果发送到指定进程。

参见 :ref:`paddle.distributed.reduce<cn_api_paddle_distributed_reduce>`。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **tensor** (Tensor) - 输入的 tensor。在目标进程上，返回结果将保存到该 tensor 中。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - **dst** (int，可选) - 目标进程的 rank，规约结果将发送到该进程。默认为 0，即结果将发送到 rank=0 的进程。
    - **op** (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD|ReduceOp.AVG，可选) - 归约的操作类型，包括求和、取最大值、取最小值、求乘积和求平均值。默认为求和。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **sync_op** (bool，可选) - 该操作是否为同步操作。默认为 True，即同步操作。
    - **use_calc_stream** (bool，可选) - 该操作是否在计算流上进行。默认为 False，即不在计算流上进行。该参数旨在提高同步操作的性能，请确保在充分了解其含义的情况下调整该参数的值。

返回
:::::::::
``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

代码示例
:::::::::
COPY-FROM: paddle.distributed.communication.stream.reduce
