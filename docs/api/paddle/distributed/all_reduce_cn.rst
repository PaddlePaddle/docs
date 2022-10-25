.. _cn_api_distributed_all_reduce:

all_reduce
-------------------------------


.. py:function:: paddle.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=0)

进程组内所有进程的指定 tensor 进行归约操作，并返回给所有进程归约的结果。
如下图所示，4 个 GPU 分别开启 4 个进程，每张卡上的数据用卡号代表，规约操作为求和，
经过 all_reduce 算子后，每张卡都会拥有所有卡数据的总和。

.. image:: ./img/allreduce.png
  :width: 800
  :alt: all_reduce
  :align: center

参数
:::::::::
    - **tensor** (Tensor) - 操作的输入 Tensor，同时也会将归约结果返回至此 Tensor 中。Tensor 的数据类型为：float16、float32、float64、int32、int64。
    - **op** (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD，可选) - 归约的具体操作，比如求和，取最大值，取最小值和求乘积，默认为求和归约。
    - **group** (int，可选) - 工作的进程组编号，默认为 0。

返回
:::::::::
无

代码示例
:::::::::
COPY-FROM: paddle.distributed.all_reduce
