.. _cn_api_distributed_scatter:

scatter
-------------------------------


.. py:function:: paddle.distributed.scatter(tensor, tensor_list=None, src=0, group=0)

进程组内指定进程源的tensor列表分发到其他所有进程中。

参数
:::::::::
    - tensor (Tensor) - 操作的输出Tensor。Tensor的数据类型为：float16、float32、float64、int32、int64。
    - tensor_list (list，可选) - 操作的输入Tensor列表，默认为None。列表中的每个元素均为Tensor，每个Tensor的数据类型为：float16、float32、float64、int32、int64。
    - src (int，可选) - 操作的源进程号，该进程号的Tensor列表将分发到其他进程中。默认为0。
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
        if paddle.distributed.ParallelEnv().local_rank == 0:
            np_data1 = np.array([7, 8, 9])
            np_data2 = np.array([10, 11, 12])
        else:
            np_data1 = np.array([1, 2, 3])
            np_data2 = np.array([4, 5, 6])
        data1 = paddle.to_tensor(np_data1)
        data2 = paddle.to_tensor(np_data2)
        if paddle.distributed.ParallelEnv().local_rank == 0:
            paddle.distributed.scatter(data1, src=1)
        else:
            paddle.distributed.scatter(data1, tensor_list=[data1, data2], src=1)
        out = data1.numpy()
