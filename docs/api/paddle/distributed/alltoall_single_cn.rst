.. _cn_api_distributed_alltoall_single:

alltoall_single
-------------------------------


.. py:function:: alltoall_single(in_tensor, out_tensor, in_split_sizes=None, out_split_sizes=None, group=None, use_calc_stream=True)

将输入的 tensor 分发到所有进程，并将接收到的 tensor 聚合到 out_tensor 中。

参数
:::::::::
    - in_tensor (Tensor): 输入的 tensor，其数据类型必须是 float16、float32、float64、int32、int64、int8、uint8、bool、bfloat16。
    - out_tensor (Tensor): 输出的 tensor，其数据类型与输入的 tensor 一致。
    - in_split_sizes (list[int]，可选): 对 in_tensor 的 dim[0] 进行切分的大小。若该参数未指定，in_tensor 将被均匀切分到各个进程中（需要确保 in_tensor 的大小能够被组中的进程数整除）。默认值：None。
    - out_split_sizes (list[int]，可选): 对 out_tensor 的 dim[0] 进行切分的大小。若该参数未指定，out_tensor 将均匀地聚合来自各个进程的数据（需要确保 out_tensor 的大小能够被组中的进程数整除）。默认值：None。
    - use_calc_stream (bool，可选) - 标识使用计算流（若为 True）还是通信流。默认值：True。

返回
:::::::::
若 use_calc_stream=True，无返回值；若 use_calc_stream=False，返回一个 Task。

代码示例
:::::::::
COPY-FROM: paddle.distributed.alltoall_single
