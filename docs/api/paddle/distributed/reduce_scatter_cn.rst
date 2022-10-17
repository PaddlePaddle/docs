.. _cn_api_paddle_distributed_reduce_scatter:

reduce_scatter
-------------------------------


.. py:function:: paddle.distributed.reduce_scatter(tensor, tensor_list, op=ReduceOp.SUM, group=None, use_calc_stream=True)
规约，然后将张量列表分散到组中的所有进程上

参数
:::::::::
    - **tensor** (Tensor) - 输出的张量。
    - **tensor_list** (list[Tensor]) - 归约和切分的张量列表。
    - **op** (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD) - 操作类型，默认 ReduceOp.SUM。
    - **group** (Group，可选) - 通信组；如果是 None，则使用默认通信组。
    - **use_calc_stream** (bool，可选) - 决定是在计算流还是通信流上做该通信操作；默认为 True，表示在计算流。


返回
:::::::::
返回 Task。

注意
:::::::::
当前只支持动态图

代码示例
:::::::::
COPY-FROM: paddle.distributed.reduce_scatter
