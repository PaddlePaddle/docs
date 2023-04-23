.. _cn_api_distributed_stream_scatter:

scatter
-------------------------------


.. py:function:: paddle.distributed.stream.scatter(tensor, tensor_or_tensor_list=None, src=0, group=None, sync_op=True, use_calc_stream=False)

将一组来自指定进程的 tensor 分发到每个进程。

参见 :ref:`paddle.distributed.scatter<cn_api_distributed_scatter>`。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **tensor** (Tensor) - 用于接收数据的 tensor，数据类型必须与输入保持一致。
    - **tensor_or_tensor_list** (Tensor|List[Tensor]，可选) - 待分发的数据，可以是一个 tensor 或 tensor 列表。若为 tensor，该 tensor 的大小必须与所有用于接收数据的 tensor 沿 dim[0] 拼接后的大小相同。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。默认为 None，因为非目标进程上的该参数将被忽略。
    - **src** (int，可选) - 目标进程的 rank，该进程的 tensor 列表将被分发到其他进程中。默认为 0，即分发 rank=0 的进程上的 tensor 列表。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **sync_op** (bool，可选) - 该操作是否为同步操作。默认为 True，即同步操作。
    - **use_calc_stream** (bool，可选) - 该操作是否在计算流上进行。默认为 False，即不在计算流上进行。该参数旨在提高同步操作的性能，请确保在充分了解其含义的情况下调整该参数的值。

返回
:::::::::
``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

代码示例
:::::::::
COPY-FROM: paddle.distributed.communication.stream.scatter
