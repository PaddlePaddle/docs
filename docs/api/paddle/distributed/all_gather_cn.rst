.. _cn_api_distributed_all_gather:

all_gather
-------------------------------


.. py:function:: paddle.distributed.all_gather(tensor_list, tensor, group=0)

进程组内所有进程的指定tensor进行聚合操作，并返回给所有进程聚合的结果。

参数
:::::::::
    - tensor_list (list) - 操作的输出Tensor列表。列表中的每个元素均为Tensor，每个Tensor的数据类型为：float16、float32、float64、int32、int64。
    - tensor (Tensor) - 操作的输入Tensor。Tensor的数据类型为：float16、float32、float64、int32、int64。
    - group (int，可选) - 工作的进程组编号，默认为0。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import numpy as np
        import paddle
        from paddle.distributed import init_parallel_env

        paddle.set_device('gpu:%d'%paddle.distributed.ParallelEnv().dev_id)
        init_parallel_env()
        tensor_list = []
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data1 = np.array([[4, 5, 6], [4, 5, 6]])
            np_data2 = np.array([[4, 5, 6], [4, 5, 6]])
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.all_gather(tensor_list, data1)
        else:
            np_data1 = np.array([[1, 2, 3], [1, 2, 3]])
            np_data2 = np.array([[1, 2, 3], [1, 2, 3]])
            data1 = paddle.to_tensor(np_data1)
            data2 = paddle.to_tensor(np_data2)
            paddle.distributed.all_gather(tensor_list, data2)
