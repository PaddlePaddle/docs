.. _cn_api_paddle_distributed_batch_isend_irecv:

batch_isend_irecv
-------------------------------


.. py:function:: paddle.distributed.batch_isend_irecv(p2p_op_list) 
异步发送或接收一批张量并返回请求列表


参数
:::::::::
    - p2p_op_list – 点对点操作列表（每个操作符的类型为 ``paddle.distributed.P2POp``）。列表中 ``isend``/ ``irecv`` 的顺序需要与远程端对应的 ``isend`` / ``irecv`` 匹配。
 
返回
:::::::::
返回Task列表。

代码示例
:::::::::
COPY-FROM: paddle.distributed.batch_isend_irecv