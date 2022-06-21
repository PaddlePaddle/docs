.. _cn_api_distributed_send:

send
-------------------------------


.. py:function:: paddle.distributed.send(tensor, dst=0, group=None, use_calc_stream=True)

发送tensor到指定接收者。

参数
:::::::::
    - tensor (Tensor) - 需要发送的Tensor。数据类型为：float16、float32、float64、int32、int64。
    - dst (int) - 接收者的标识符。
    - group (Group，可选) - new_group返回的Group实例，或者设置为None表示默认地全局组。默认值：None。
    - use_calc_stream (bool，可选) - 标识使用计算流还是通信流。默认值：True。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.send