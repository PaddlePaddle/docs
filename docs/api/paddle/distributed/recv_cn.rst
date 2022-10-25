.. _cn_api_distributed_recv:

recv
-------------------------------


.. py:function:: paddle.distributed.recv(tensor, src=0, group=None, use_calc_stream=True)

发送 tensor 到指定接收者。

参数
:::::::::
    - **tensor** (Tensor) - 接收数据的 Tensor。数据类型为：float16、float32、float64、int32、int64。
    - **src** (int) - 发送者的标识符。
    - **group** (Group，可选) - new_group 返回的 Group 实例，或者设置为 None 表示默认地全局组。默认值：None。
    - **use_calc_stream** (bool，可选) - 标识使用计算流还是通信流。默认值：True。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.recv
