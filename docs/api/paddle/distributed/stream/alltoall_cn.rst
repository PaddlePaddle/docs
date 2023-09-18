.. _cn_api_paddle_distributed_stream_alltoall:

alltoall
-------------------------------


.. py:function:: paddle.distributed.stream.alltoall(out_tensor_or_tensor_list, in_tensor_or_tensor_list, group=None, sync_op=True, use_calc_stream=False)

将一个或一组 tensor 分发到每个进程，随后在每个进程上聚合分发结果。

参见 :ref:`paddle.distributed.alltoall<cn_api_paddle_distributed_alltoall>`。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **out_tensor_or_tensor_list** (Tensor|List[Tensor]) - 用于保存操作结果。若输入数据为 tensor，该参数必须为 tensor，且大小与所有输入的 tensor 沿 dim[0] 拼接后的大小相同。若输入数据为 tensor 列表，该参数必须为 tensor 列表，其中每个 tensor 的数据类型必须与输入的 tensor 保持一致。
    - **in_tensor_or_tensor_list** (Tensor|List[Tensor]) - 输入的数据，可以是一个 tensor 或 tensor 列表。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **sync_op** (bool，可选) - 该操作是否为同步操作。默认为 True，即同步操作。
    - **use_calc_stream** (bool，可选) - 该操作是否在计算流上进行。默认为 False，即不在计算流上进行。该参数旨在提高同步操作的性能，请确保在充分了解其含义的情况下调整该参数的值。

返回
:::::::::
``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

代码示例
:::::::::
COPY-FROM: paddle.distributed.communication.stream.alltoall
