.. _cn_api_distributed_ReduceOp:

ReduceOp
-------------------------------

指定规约类操作的逐元素操作类型，需要是下述值之一：

    ReduceOp.SUM

    ReduceOp.MAX

    ReduceOp.MIN

    ReduceOp.PROD

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
        paddle.distributed.all_reduce(data, op=ReduceOp.SUM)
        out = data.numpy()
        # [[5, 7, 9], [5, 7, 9]]
