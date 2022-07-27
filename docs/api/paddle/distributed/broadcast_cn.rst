.. _cn_api_distributed_broadcast:

broadcast
-------------------------------


.. py:function:: paddle.distributed.broadcast(tensor, src, group=0)

广播一个Tensor给其他所有进程。
如下图所示，4个GPU分别开启4个进程，GPU0卡拥有数据，经过broadcast算子后，会将这个数据传播到所有卡上。

.. image:: ./img/broadcast.png
  :width: 800
  :alt: broadcast
  :align: center

参数
:::::::::
    - tensor (Tensor) - 如果当前进程编号是源，那么这个Tensor变量将被发送给其他进程，否则这个Tensor将接收源发送过来的数据。Tensor的数据类型为：float16、float32、float64、int32、int64。
    - src (int) - 发送源的进程编号。
    - group (int，可选) - 工作的进程组编号，默认为0。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.broadcast
