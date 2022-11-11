.. _cn_api_distributed_stream_alltoall_single:

alltoall_single
-------------------------------


.. py:function:: paddle.distributed.stream.alltoall_single(out_tensor, in_tensor, out_split_sizes=None, in_split_sizes=None, group=None, sync_op=True, use_calc_stream=False)

将一个 tensor 分发到每个进程，随后在每个进程上聚合分发结果。与 ``alltoall`` 相比，可以更精细地控制分发过程。

参见 :ref:`paddle.distributed.alltoall_single<cn_api_distributed_alltoall_single>`。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **out_tensor** (Tensor): 用于保存操作结果的 tensor，数据类型必须与输入的 tensor 保持一致。
    - **in_tensor** (Tensor): 输入的 tensor。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - **out_split_sizes** (List[int]，可选): 对 out_tensor 的 dim[0] 进行切分的大小。默认为 None，即 out_tensor 将均匀地聚合各个进程的数据（需要确保 out_tensor 的大小能够被组中的进程数整除）。
    - **in_split_sizes** (List[int]，可选): 对 in_tensor 的 dim[0] 进行切分的大小。默认为 None，即将 in_tensor 均匀地分发到各个进程中（需要确保 in_tensor 的大小能够被组中的进程数整除）。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **sync_op** (bool，可选) - 该操作是否为同步操作。默认为 True，即同步操作。
    - **use_calc_stream** (bool，可选) - 该操作是否在计算流上进行。默认为 False，即不在计算流上进行。该参数旨在提高同步操作的性能，请确保在充分了解其含义的情况下调整该参数的值。

返回
:::::::::
``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

代码示例
:::::::::
COPY-FROM: paddle.distributed.communication.stream.alltoall_single
