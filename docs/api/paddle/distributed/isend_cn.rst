.. _cn_api_paddle_distributed_isend:

isend
-------------------------------


.. py:function:: paddle.distributed.isend(tensor, dst, group=None)
异步的将 ``tensor`` 发送到指定的 rank 进程上。

参数
:::::::::
    - **tensor** (Tensor) - 要发送的张量。其数据类型应为 float16、float32、float64、int32 或 int64。
    - **dst** (int) - 目标节点的全局 rank 号。
    - **group** (Group，可选) - new_group 返回的 Group 实例，或者设置为 None 表示默认的全局组。默认值：None。


返回
:::::::::
返回 Task。


注意
:::::::::
当前只支持动态图

代码示例
:::::::::
COPY-FROM: paddle.distributed.isend
