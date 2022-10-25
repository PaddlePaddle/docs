.. _cn_api_paddle_distributed_irecv:

irecv
-------------------------------


.. py:function:: paddle.distributed.irecv(tensor, src=None, group=None)
异步接受发送来的 tensor。

参数
:::::::::
    - **tensor** (Tensor) - 要接受的张量。其数据类型应为 float16、float32、float64、int32 或 int64。
    - **src** (int) - 接受节点的全局 rank 号。
    - **group** (Group，可选) - new_group 返回的 Group 实例，或者设置为 None 表示默认的全局组。默认值：None。


返回
:::::::::
返回 Task。

注意
:::::::::
当前只支持动态图

代码示例
:::::::::
COPY-FROM: paddle.distributed.irecv
