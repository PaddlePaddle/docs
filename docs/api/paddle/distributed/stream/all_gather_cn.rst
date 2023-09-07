.. _cn_api_paddle_distributed_stream_all_gather:

all_gather
-------------------------------


.. py:function:: paddle.distributed.stream.all_gather(tensor_or_tensor_list, tensor, group=None, sync_op=True, use_calc_stream=False)

聚合进程组内的指定 tensor，随后将聚合结果发送到每个进程。

参见 :ref:`paddle.distributed.all_gather<cn_api_paddle_distributed_all_gather>`。

.. note::
  该 API 只支持动态图模式。

参数
:::::::::
    - **tensor_or_tensor_list** (Tensor|List[Tensor]) - 用于保存聚合结果。若为 tensor，该 tensor 的大小必须与所有待聚合的 tensor 沿 dim[0] 拼接后的大小相同。若为 tensor 列表，其中每个 tensor 的数据类型必须与输入的 tensor 保持一致。
    - **tensor** (Tensor) - 待聚合的 tensor。支持的数据类型包括：float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - **group** (Group，可选) - 执行该操作的进程组实例（通过 ``new_group`` 创建）。默认为 None，即使用全局默认进程组。
    - **sync_op** (bool，可选) - 该操作是否为同步操作。默认为 True，即同步操作。
    - **use_calc_stream** (bool，可选) - 该操作是否在计算流上进行。默认为 False，即不在计算流上进行。该参数旨在提高同步操作的性能，请确保在充分了解其含义的情况下调整该参数的值。

返回
:::::::::
``Task``。通过 ``Task``，可以查看异步操作的执行状态以及等待异步操作的结果。

代码示例
:::::::::
COPY-FROM: paddle.distributed.communication.stream.all_gather
