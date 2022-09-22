.. _cn_api_distributed_reduce:

reduce
-------------------------------


.. py:function:: paddle.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=0)

进程组内所有进程的指定 tensor 进行归约操作，并返回给所有进程归约的结果。
如下图所示，4 个 GPU 分别开启 4 个进程，每张卡上的数据用卡号代表，reduce 的目标是第 0 张卡，
规约操作是求和，经过 reduce 操作后，第 0 张卡会得到所有卡数据的总和。

.. image:: ./img/reduce.png
  :width: 800
  :alt: reduce
  :align: center

参数
:::::::::
    - **tensor** (Tensor) - 操作的输入 Tensor，结果返回至目标进程号的 Tensor 中。Tensor 的数据类型为：float16、float32、float64、int32、int64。
    - **dst** (int) - 返回操作结果的目标进程编号。
    - **op** (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD，可选) - 归约的具体操作，比如求和，取最大值，取最小值和求乘积，默认为求和归约。
    - **group** (int，可选) - 工作的进程组编号，默认为 0。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.reduce
