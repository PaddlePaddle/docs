.. _cn_api_distributed_stream_reduce_scatter:

reduce_scatter
-------------------------------


.. py:function:: paddle.distributed.stream.reduce_scatter(tensor, tensor_or_tensor_list, op=ReduceOp.SUM, group=None, sync_op=True, use_calc_stream=False)

规约一组 tensor，随后将规约结果分发到每个进程。

参见 :ref:`paddle.distributed.reduce_scatter<cn_api_distributed_reduce_scatter>`。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **tensor** (Tensor) – 用于接收数据的 tensor，数据类型必须与输入保持一致。
    - **tensor_or_tensor_list** (Tensor|List[Tensor]) - 输入的数据，可以是一个 tensor 或 tensor 列表。若为 tensor，该 tensor 的大小必须与所有用于接收数据的 tensor 沿 dim[0] 拼接后的大小相同。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - **op** (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD，可选) - 归约的操作类型，包括求和、取最大值、取最小值和求乘积。默认为求和。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **sync_op** (bool，可选) - 该操作是否为同步操作。默认为 True，即同步操作。
    - **use_calc_stream** (bool，可选) - 该操作是否在计算流上进行。默认为 False，即不在计算流上进行。该参数旨在提高同步操作的性能，请确保在充分了解其含义的情况下调整该参数的值。


返回
:::::::::
``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

代码示例
:::::::::
COPY-FROM: paddle.distributed.communication.stream.reduce_scatter
