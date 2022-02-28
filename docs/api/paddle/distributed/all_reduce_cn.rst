.. _cn_api_distributed_all_reduce:

all_reduce
-------------------------------


.. py:function:: paddle.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=0)

进程组内所有进程的指定tensor进行归约操作，并返回给所有进程归约的结果。
如下图所示，4个GPU分别开启4个进程，每张卡上的数据用卡号代表，规约操作为求和，
经过all_reduce算子后，每张卡都会拥有所有卡数据的总和。

.. image:: ./img/allreduce.png
  :width: 800
  :alt: all_reduce
  :align: center

参数
:::::::::
    - tensor (Tensor) - 操作的输入Tensor，同时也会将归约结果返回至此Tensor中。Tensor的数据类型为：float16、float32、float64、int32、int64。
    - op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD，可选) - 归约的具体操作，比如求和，取最大值，取最小值和求乘积，默认为求和归约。
    - group (int，可选) - 工作的进程组编号，默认为0。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import numpy as np
        import paddle
        from paddle.distributed import ReduceOp
        from paddle.distributed import init_parallel_env

        paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
        init_parallel_env()
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data = np.array([[4, 5, 6], [4, 5, 6]])
        else:
            np_data = np.array([[1, 2, 3], [1, 2, 3]])
        data = paddle.to_tensor(np_data)
        paddle.distributed.all_reduce(data)
        out = data.numpy()
        # [[5, 7, 9], [5, 7, 9]]
