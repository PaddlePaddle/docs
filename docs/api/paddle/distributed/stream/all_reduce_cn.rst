.. _cn_api_distributed_stream_all_reduce:

all_reduce
-------------------------------


.. py:function:: paddle.distributed.stream.all_reduce(tensor, op=ReduceOp.SUM, group=None, sync_op=True, use_calc_stream=False)

和 paddle.distributed.all_reduce 的语义相同，进程组内的所有进程在指定 tensor 上所特定的归约操作 （如 sum，reduce 等）。


参数
:::::::::
    - tensor (Tensor) - 操作的输入 Tensor，同时也会将归约结果返回至此 Tensor 中。Tensor 的数据类型为：float16、float32、float64、int32、int64。
    - op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.MIN|ReduceOp.PROD，可选) - 归约的具体操作，比如求和，取最大值，取最小值和求乘积，默认为求和归约。
    - group (Group，可选) - 在指定通信组中做通信，是可选参数，默认使用全局通信组。
    - sync_op (bool, 可选) - 表示做同步或者异步通信，默认进行同步通信。
    - use_calc_stream (bool, 可选) - 表示是否在计算流上做通信，默认在通信流上进行通信。

返回
:::::::::
返回一个 task 对象。

警告
:::::::::
该 API 目前只支持在动态图下使用

代码示例
:::::::::
COPY-FROM: paddle.distributed.communication.stream.all_reduce
