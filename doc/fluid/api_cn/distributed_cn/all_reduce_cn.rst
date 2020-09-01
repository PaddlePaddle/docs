.. _cn_api_distributed_all_reduce:

all_reduce
-------------------------------


.. py:function:: paddle.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=0)

进程组内所有进程的指定tensor进行归约操作，并返回给所有进程归约的结果。

参数
:::::::::
    - tensor (Tensor) - 操作的输入Tensor，同时也会将归约结果返回至此Tensor中。Tensor的数据类型为：float32、float64、int32、int64。
    - op (ReduceOp.SUM|ReduceOp.MAX|ReduceOp.Min|ReduceOp.PROD，可选) - 归约的具体操作，比如求和，取最大值，取最小值和求乘积，默认为求和归约。
    - group (int，可选) - 工作的进程组编号，默认为0。

返回
:::::::::
无

代码示例
:::::::::
.. code-block:: python

        import paddle
        from paddle.distributed import ReduceOp
        import paddle.prepare_context as prepare_context

        paddle.disable_static()
        paddle.set_device('gpu:%d'%paddle.ParallelEnv().dev_id)
        prepare_context()
        if paddle.ParallelEnv().local_rank == 0:
            np_data = np.array([[4, 5, 6], [4, 5, 6]])
        else:
            np_data = np.array([[1, 2, 3], [1, 2, 3]])
        data = paddle.to_tensor(np_data)
        paddle.distributed.all_reduce(data)
        out = data.numpy()
